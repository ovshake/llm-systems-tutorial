from typing import Any, cast 
import torch 
from loguru import logger 
from torch import nn 
from torch.distributed._composable.fsdp import FSDPModule 
from torch.distributed._composable.replicate import replicate 
from torch.distributed.device_mesh import DeviceMesh 
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard 
from torch.distributed.tensor import Shard 



def apply_ddp(
    model: nn.Module, 
    dp_mesh: DeviceMesh, 
    enable_compile: bool, 
    enabled_compiled_autograd: bool
):
    if enable_compile:
        if enabled_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = (
                "ddp_optimizer"
            )
    
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100) 

    logger.info(f"Applied DDP to the model")


def apply_fsdp(
    model: nn.Module, 
    dp_mesh: DeviceMesh, 
    param_dtype: torch.dtype, 
    reduce_dtype: torch.dtype, 
    pp_enabled: bool, 
    reshard_after_forward_policy: str = "default", 
    ep_degree: int = 1, 
    dp_replicate_efsdp_mesh: DeviceMesh | None = None,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, 
        reduce_dtype=reduce_dtype, 
        cast_forward_inputs=False, 
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy} 
    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True 
        case "never": 
            reshard_after_forward = False 
        case "default":
            reshard_after_forward = not pp_enabled 
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}"
            )
    
    tok_embeddings: Any = model.tok_embeddings
    layers = cast(nn.ModuleDict, model.layers) 
    norm: Any = model.norm 
    output: Any = model.output 

    if tok_embeddings is not None:
        fully_shard(
            tok_embeddings, 
            **fsdp_config, 
            reshard_after_forward=reshard_after_forward,
        )
    
    for transformer_block in layer.values():
        # Skipping MoE stuff for now 

        fully_shard(
            transformer_block, 
            **fsdp_config, 
            reshard_after_forward=reshard_after_forward
        )


    if norm is not None and output is not None:
        fully_shard(
            [norm, output], 
            **fsdp_config, 
            reshard_after_forward=reshard_after_forward_policy == "always"
        )
    
    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model) 

    if ep_degree == 1:
        return  
    
    # forward 
    transformer_blocks: list[Any] = list(layers.values()) 
    next_transformer_blocks = transformer_blocks[1:] + [None] 
    if tok_embeddings is not None and len(layers) > 0:
        tok_embeddings.set_modules_to_forward_fetch([transformer_blocks[0]])
    
    for transformer_block, next_transformer_block in zip(transformer_blocks, next_transformer_blocks):
        if next_transformer_block is not None:
            if next_transformer_block.moe.enabled:
                transformer_block.set_modules_to_forward_fetch([next_transformer_block, next_transformer_block.moe.experts]) 
            else:
                transformer_block.set_modules_to_forward_fetch([next_transformer_block]) 
        elif norm is not None and output is not None:
            transformer_block.set_modules_to_forward_fetch([norm, output]) 
    
    # backward
    reversed_transformer_blocks: list[Any] = reversed(list(layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None] 

    if norm is not None and output is not None and len(reversed_transformer_blocks) > 0:
        output.set_modules_to_forward_fetch([reversed_transformer_blocks[0]])
    
    for transformer_block, prev_transformer_block in zip(reversed_transformer_blocks, prev_transformer_blocks):
        if prev_transformer_block is not None:
            if prev_transformer_block.moe.enabled:
                transformer_block.set_modules_to_forward_fetch([prev_transformer_block, prev_transformer_block.moe.experts]) 
            else:
                transformer_block.set_modules_to_forward_fetch([prev_transformer_block]) 
        
        elif tok_embeddings is not None:
            transformer_block.set_modules_to_forward_fetch([tok_embeddings])
            


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


