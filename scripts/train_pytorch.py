"""
PI0/PI05 的 PyTorch 训练入口（支持多 GPU / 多机 DDP）

该脚本与 JAX 训练脚本（scripts/train.py）行为保持一致，但完全使用 PyTorch
并基于 PI0Pytorch 模型与现有配置 / 数据管线运行。

Usage
单 GPU：
    python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
    示例：
    python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
    python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # 从最新检查点恢复
多 GPU（单机）：
    torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
    示例：
    torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
    torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
多机训练：
	torchrun \
        --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
        --master_addr=<master_ip> --master_port=<port> \
        scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
"""

import dataclasses  # 用于将配置 dataclass 转为字典
import gc  # 显存/内存清理辅助
import logging  # 日志输出
import os  # 读取环境变量、路径拼接
import platform  # 记录运行机器信息
import shutil  # 文件/目录复制与删除
import time  # 计时与时间戳

import jax  # 复用 JAX 的树结构工具（仅用于数据转移）
import numpy as np  # 数值计算
import safetensors.torch  # 安全的权重保存/加载
import torch  # PyTorch 主库
import torch.distributed as dist  # 分布式训练（DDP）
import torch.nn.parallel  # DDP 包装类
import tqdm  # 进度条
import wandb  # 实验记录

import openpi.models.pi0_config  # PI0 配置结构
import openpi.models_pytorch.pi0_pytorch  # PyTorch 版 PI0 模型
import openpi.shared.normalize as _normalize  # 归一化统计保存/读取
import openpi.training.config as _config  # 训练配置解析
import openpi.training.data_loader as _data  # 统一数据加载器


def init_logging():
    """初始化日志格式与输出渠道。"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # 设置日志格式：时间 + 级别 + 消息 + 进程/文件/行号
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()  # 获取根 logger
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """初始化 W&B 记录（主进程调用）。"""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir  # 检查点目录（包含 wandb_id.txt）
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        # 恢复训练：读取此前 run_id 并强制 resume
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新训练：创建新的 run 并记录配置
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    """初始化分布式训练环境并返回 (use_ddp, local_rank, device)。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))  # 总进程数（默认 1）
    use_ddp = world_size > 1  # world_size>1 即启用 DDP
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"  # GPU 用 NCCL，CPU 用 GLOO
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        # 开启 DDP 调试日志，方便排查 rank 之间的初始化或通信问题
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))  # 进程在本机的 rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")  # 绑定到对应 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    """销毁分布式进程组，确保所有进程同步退出。"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # 先同步，避免提前退出
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    """设置随机种子，保证多进程可复现。"""
    torch.manual_seed(seed + local_rank)  # CPU 侧随机
    np.random.seed(seed + local_rank)  # NumPy 随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)  # GPU 侧随机


def build_datasets(config: _config.TrainConfig):
    """构建数据加载器，并返回 (loader, data_config)。"""
    # 使用统一数据加载器（框架设为 PyTorch）
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()  # 同时返回数据配置（含归一化统计）


def get_model_state_dict(model):
    """获取模型 state_dict（兼容 DDP 包装）。"""
    return (
        model.module.state_dict()  # DDP 包装后参数在 module 内
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()  # 单卡直接取
    )


def get_model_parameters(model):
    """获取模型参数（兼容 DDP 包装）。"""
    return (
        model.module.parameters()  # DDP 包装后参数在 module 内
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()  # 单卡直接取
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """保存检查点（模型权重 + 优化器状态 + 元数据）。"""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving 
        # 创建最终检查点目录与临时目录
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型权重（safetensors 支持共享张量）
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # 保存优化器状态（PyTorch 格式）
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # 保存元数据（避免完整配置带来的 JAX/Flax 兼容问题）
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # 保存归一化统计（若存在）
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # 原子替换：避免半写入状态
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # 记录保存到 W&B
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """加载最新检查点并返回 global_step。"""
    # 扫描目录下所有数字命名的检查点目录
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)  # 选择最新 step
    ckpt_dir = checkpoint_dir / f"{latest_step}"  # 对应检查点路径

    # 加载前尽量释放显存/内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # 加载模型权重（带错误处理）
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # 加载优化器状态（带错误处理）
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)  # 恢复优化器内部状态
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # 加载元数据（包含 global_step）
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """从检查点目录中获取最新的 step。"""
    # 与 load_checkpoint 相同的目录扫描逻辑
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """记录当前 GPU 显存使用情况（仅 CUDA 可用时）。"""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # 获取更详细的显存统计
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # 如在 DDP 中，补充 rank/world_size 信息
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    """训练主循环（支持 DDP）。"""
    use_ddp, local_rank, device = setup_ddp()  # 初始化 DDP 并获取设备
    is_main = (not use_ddp) or (dist.get_rank() == 0)  # 主进程用于日志/保存
    set_seed(config.seed, local_rank)  # 为每个进程设置种子

    # 初始化检查点目录与 W&B
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 使用统一数据加载器，计算每张卡的有效 batch size
    # N 张卡时，每张卡拿 batch_size/N，总体 batch size 仍为 batch_size，保持与单机单卡同等的全局 batch
    world_size = torch.distributed.get_world_size() if use_ddp else 1  # DDP 总进程数
    effective_batch_size = config.batch_size // world_size  # 每卡 batch size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)  # 构建训练数据加载器与配置

    # 仅在主进程记录样例图像到 W&B（避免重复）
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []  # 保存前 5 个样例
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]  # 从任意视角取 batch size
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat(
                [img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1
            )
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)  # 记录到 W&B

        # Clear sample batch from memory aggressively（样例批次只用一次，尽快释放，避免与正式 loader 竞争显存/内存）
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # 构建模型（兼容 dataclass / Pi0Config）
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)  # 实例化并移到设备

    # 启用梯度检查点以降低显存占用（牺牲少量计算，换取激活不保留）
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # 大规模训练时启用显存优化选项
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # DDP 包装
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # 若指定预训练权重路径，则加载用于微调（覆盖默认初始化）
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # 优化器与学习率计划
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # 若恢复训练，加载最新检查点
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        """Cosine decay 学习率计划（含 warmup）。"""
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            # 前 warmup_steps 线性升温，起点略低于 peak_lr，确保首步不过大
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay：从 peak_lr 平滑衰减到 end_lr（余弦形状，末尾平台）
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()  # 进入训练模式
    start_time = time.time()  # 用于计算日志间隔耗时
    infos = []  # 收集 log_interval 内的统计信息
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # 训练循环：直到达到 num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # DDP 下为采样器设置 epoch
        if use_ddp and hasattr(loader, "set_epoch"):
            # set_epoch 让 DistributedSampler 在每个 epoch 重新打乱，避免各 rank 采到重复/偏移数据
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # 统一数据加载器返回 (observation, actions)
            # observation 是嵌套结构（含多视角图像与状态），用 jax.tree.map 递归调用 .to(device)
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            # actions 先转 float32（模型期望）再移动到设备
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR：每步更新 param_groups，避免手写 scheduler 对齐问题
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # 前向传播：模型返回可迭代或张量的 loss 列表
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()  # 多段 loss 取平均作为标量

            # 反向传播：累积梯度
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # 梯度裁剪：限制梯度 L2 范数，防止梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # 参数更新
            optim.step()
            optim.zero_grad(set_to_none=True)

            # 更激进地清理梯度，降低显存峰值（DDP 下长序列可明显节省内存）
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # 收集指标
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            # 到达 log_interval 时输出日志并写入 W&B（聚合 interval 内平均值，避免每步刷日志）
            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # 计算 log_interval 内的平均指标
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # 写入 W&B：仅主进程写，减少 API 调用
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()  # 重置计时器
                infos = []  # 重置统计缓存

            global_step += 1  # 训练步数 +1
            # 保存检查点
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # 更新进度条
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # 关闭进度条
    if pbar is not None:
        pbar.close()

    # 结束 W&B 运行
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    """脚本入口：初始化日志与配置后启动训练。"""
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
