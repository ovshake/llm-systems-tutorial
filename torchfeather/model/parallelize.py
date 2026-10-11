import torch 
from loguru import logger 
from torch import nn 
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Replicate, Shard 
from torch.distributed.tensor.parallel import (
    ColwiseParallel, 
    ParallelStyle, 
    PrepareModuleInput, 
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from torchfeather.config import TORCH_DTYPE_MAP, JobConfig 
from torchfeather.config.job_config import Compile as CompileConfig 
from torchfeather.distributed import NoParallel, ParallelDims 
from torchfeather.distributed.activation_checkpoint import apply_ac
from torchfeather.distributed.expert_parallel import apply_moe_ep_tp 
from torchfeather.distributed.model_parallel import apply_ddp, apply_fsdp 


# for selective activation checkpointing 

def _build_op_sac_save_list() -> set:
    """Build the set of ops whose output should be saved of selective activation checkpointing""" 
    try:
        from torch._functorch.partitioner import get_default_op_list

        save_ops = {op.default for op in get_default_op_list().compute_intensive_ops}
    except (ImportError, AttributeError) as e:
        save_ops = [
            torch.ops.aten.mm.default, 
            torch.ops.aten._scaled_dot_product_efficient_attention.default, 
            torch.ops.aten._scaled_dot_product_flash_attention.default,
        ]
    
    extra_ops = [
        torch.ops.aten.linear.default, 
        torch.ops.aten._scaled_dot_product_cudnn_attention.default, 
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default, 
        torch.ops._c10d_functional.all_to_all_single.default, 
        torch.ops.aten.max.default, 
    ]

    for op in extra_ops:
        if op is not None:
            save_ops.add(op) 
    
    # FlexAttention higher order op (may not exist in all PyTorch versions) 
    if hasattr(torch._higher_order_ops, "flex_attention"):
        save_ops.add(torch._higher_order_ops.flex_attention)
    

    return save_ops


_ops_sac_save_list = _build_op_sac_save_list() 

def parallelize_deepseekv3(
    model: nn.Module, 
    parallel_dims: ParallelDims, 
    job_config: JobConfig,
)

    assert job_config.training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
    Sequence length {job_config.training.seq_len} must be divisible by 
    the product of the TP degree ({parallel_dims.tp}) and 2 * cp degree ({parallel_dims.cp}) 
    """

    if parallel_dims.tp_enabled:
        tp = parallel_dims.tp 
        n_heads = model.model_args.n_heads 
        if n_heads % tp != 0:
            raise ValueError(
                f"number of heads should be divisible by TP degree"
            )
    
    apply_non_moe_tp(
        model, 
        parallel_dims.get_mesh("tp"), 
        loss_parallel=not job_config.parallelism.disable_loss_parallel
    )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model, 
            job_config.activation_checkpoint, 
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_ops_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    
    if model_compile_enabled:
        apply_compile(model, job_config.compile)


    dp_mesh: DeviceMesh | None = None 
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # Apply FSDP or HSDP potentially with context parallel 
        if parallel_dims.dp_replicate_enabled:
            dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
        else:
            dp_mesh = parallel_dims.get_mesh(["fsdp"]) 
    

        # Mesh for FSDP sharding of the MoE in the EP region 

        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_replicate_efsdp_mesh = parallel_dims.get_mesh(["dp_replicate", "efsdp"]) 
            else:
                dp_replicate_efsdp_mesh = parallel_dims.get_mesh(["efsdp"])
        else:
            dp_replicate_efsdp_mesh = None 


    apply_fsdp(
        model,  
        dp_mesh, 
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param], 
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce], 
        pp_enabled=parallel_dims.pp_enabled,
        reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep, 
        dp_replicate_efsdp_mesh=dp_replicate_efsdp_mesh,
    )

        if parallel_dims.dp_replicate_enabled:
            logger.info(f"Applied HSDP to the model")
        else:
            logger.info(f"Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info(f"Applied Context Parallel to the model")
        
    elif parallel_dims.dp_replicate_enabled:
        if parallel_dims.world_mesh.ndim > 1:
            raise RuntimeError("DDP is not supported in > 1 D parallelism")
    
    dp_mesh = parallel_dims.world_mesh
    apply_ddp(
        model, 
        dp_mesh, 
        enable_compile=model_compile_enabled, 
        enable_compile_autograd=job_config.parallelism.enable_compile_autograd
    )

    return model 


def apply_compile(model: nn.Module, compile_config: CompileConfig):

    # NOTE: this flag is needed to avoid graph breaks for dynamic shapes in token choice MoEs 
    # torch._dynamo.config.capture_scaler_outputs = True 

    for layer_id, transformer_block in model.layers.named_children():
        full_graph = True 

        if transformer_block.moe_enabled:
            full_graph = False 
        
        transformer_block = torch.compile(
            transformer_block, 
            backend=compile_config.backend,
            full_graph=full_graph
        )

        model.layers.register_module(layer_id, transformer_block) 
    
    logger.info(f"Compiling each transformers block with torch.compile")


def apply_non_moe_tp(
    model: nn.Module, 
    tp_mesh: DeviceMesh, 
    loss_parallel: bool
):
    parallelize_module(
        model,
        tp_mesh, 
        {
            "tok_embeddings":  RowwiseParallel(
                input_layouts=Replicate(), 
                output_layouts=Shard(1) # Across the seq dim
            ), 
            "norm": SequenceParallel(), 
            "output": ColwiseParallel(
                input_layouts=Shard(1), 
                output_layouts=Shard(-1) if loss_parallel else Replicate(), 
                use_local_output=not loss_parallel
            )
        }
    )