


class Trainer(Stateful):
    job_config: JobConfig 
    parallel_dims: ParallelDims 

    tokenizer: DeepSeekV3Tokenizer
    dataloader: BseDataLoader 
    model_parts: list[torch.nn.Module] 
    loss_fn: LossFunction 
    optimizers: OptimizerContainer 
    lr_schedulers: LRSchedulerContainer
    metrics_processor: MetricsProcessor 
    checkpointer: CheckpointManager 

    device: torch.device 
    gc_handler: utils.GarbageCollection 
    train_context: Callable[..., contextlib.AbstractContextManager]
    maybe_enable_amp: contextlib.AbstractContextManager
    gradient_accumulation_steps: int 
    pp_has_first_stage: bool 
    pp_has_last_stage: bool 

    @record 
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config 

        device_module, device_type = (
            device_utils.device_module, 
            device_utils.device_type,
        )
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device) 

        # init distributed and build states 
        torch.distributed.init_process_group(
            backend='nccl', 
            timeout=timedelta(second=job_config.comm.init_timeout_delta)
        )

        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism 
        self.parallel_dims = parallel_dims * self._create_parallel_dims(
            parallelism_config, world_size
        )
        _ = parallel_dims.world_mesh 

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch") 
            dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank() 

        else:
            dp_degree, dp_rank = 1, 0
        
        # take control of garbage collections to avoid stragglers 
        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq) 

        dist_utils.set_determinism(
            parallel_dims, 
            self.device, 
            job_config.training.seed, 
            job_config.training.deterministic,
        )

        # build tokenizer and dataloader 
        self.tokenizer = DeepSeekV3Tokenizer(job_config.model.hf_assets_path) 

        self.dataloader = build_hf_dataloader(
            dp_world_size=dp_degree, 
            dp_rank=dp_rank, 
            tokenizer=self.tokenizer, 
            job_config=job_config, 

        )

        model_args = job_config.model.args 
        model_args.max_seq_len = job_config.training.seq_len 
        # build on the meta device 

        with (
            torch.device("meta"),
            device_utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            self.metrics_processor = MetricsProcessor(job_config, parallel_dims) 

        # calculate model size and flops per token 

        (
            model_param_count, 
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(f"Model Total Parameters: {model_param_count:,}")

        self.loss_fn = build_cross_entropy_loss(job_config) 

        # verify batch sizes 
        global_batch_size = job_config.training.global_batch_size 
        if global_batch_size < 0:
            global_batch_size = job_config.training.local_batch_size * dp_degree 
        
        assert global_batch_size > 0, global_batch_size

        # This is when you specify the global batch size manually 
        # It is useful for validation of your config 

        assert (
            global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0, global_batch_size
        )

        # calculate gradient accumulation step 
        self.gradient_accumulation_steps = global_batch_size // (job_config.training.local_batch_size * dp_degree)

        assert self.gradient_accumulation_steps > 0

        init_device = device_type
        buffer_device = None 

        # apply parallelisms and initializations 

        if parallel_dims.pp_enabled:
            (
                self.pp_schedule, 
                self.model_parts, 
                self.pp_has_first_stage, 
                self.pp_has_last_stage,
            ) = pipeline_llm(
                model, 
                parallel_dims, 
                job_config, 
                self.device, 
                model_args.n_layers, 
                parallelize_deepseekv3,
                self.loss_fn
            )

            # When PP is enabled, 'model' is no longer used after this point, model_parts is used instead 
            del model 
            for m in self.model_parts:
                m.to_empty(device=init_device) 
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train() 
        
        else:
            model = parallelize_deepseekv3(model, parallel_dim, job_config) 

            model.to_empty(device=init_device) 
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            
            model.train() 

            self.model_parts = [model]

        # initialize device memory monitor and get peak flops for MFU caculations 

        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name) 
        logger.info(f"Peak FLOPs used for computing MFU: {gpu_peak_flops:.3e}") 
        device_num_stats = device_memory_monitor.get_peak_stats() 

        logger.info(
            f"{device_type.upper()} memory usage for model:"
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelism to the model 
        self.optimizers = build_optimizers_with_moe_load_balancing(
            self.model_parts, job_config.optimizer, parallel_dims 
        )
        self.lr_schedulers = build_lr_schedulers(
            self.optimizers, job_config, job_config.training.steps 
        )

        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts 


        # Initialize trainier states that will be saved in the checkpoint 
        # These attributes must be initialized before checkpoint loading 
        self.step = 0 
        self.ntokens_seen = 0 

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader, 
            model_parts=self.model_parts, 
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            additional_states={"trainer_state": self}, 
            checkpoint_config=job_config.checkpoint,
            base_folder=job_config.job.dump_folder,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )

        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )

        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims, 
            job_config.training.mixed_precision_param, 
            device_type,
        )

        logger.info(
            f"Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"Sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps}, "
            f"(warmup {job_config.lr_schedulers.warmup_steps})"
        )


    def _create_parallel_dims(
        self, parallelism_config: ParallelDims, world_size: int
    ) -> ParallelDims:
        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree, 
            dp_replicate=parallelism_config.data_parallel_replicate_degree, 
            cp=parallelism_config.context_parallel_degree, 
            tp=parallelism_config.tensor_parallel_degree, 
            pp=parallelism_config.pipeline_parallel_degree, 
            ep=parallelism_config.expert_parallel_degree, 
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size
        )




if __name__ == "__main__":
    CONFIG_NAME = os.environ.get("TORCHFEATHER_CONFIG", None) 
    if CONFIG_NAME is None:
        raise ValueError("TORCFEATHER_CONFIG environment variable is not set") 
    
    trainer: Trainer | None = None 

    try:
        config = get_config(CONFIG_NAME) 
        config.job.dump_dir = f"./outputs/{CONFIG_NAME}" 
        trainer = Trainer(config) 
        trainer.train() 

    except Exception:
        if trainer:
            trainer.close() 
    
    else:
        _shutdown_after_successful_training(trainer) 