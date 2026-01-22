"""
OpenPI 模型训练脚本

此脚本实现了完整的模型训练流程，包括：
- 训练状态初始化（模型参数、优化器状态）
- 数据加载和批处理
- 分布式训练（FSDP - Fully Sharded Data Parallel）
- 检查点保存和恢复
- 实验跟踪（Weights & Biases）

主要组件：
- init_train_state: 初始化模型和优化器
- train_step: 单步训练（前向传播、反向传播、参数更新）
- main: 主训练循环
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """
    初始化自定义日志格式
    
    配置日志输出格式为：时间戳 [级别] 消息 (进程ID:文件名:行号)
    将日志级别缩写为单字母（D/I/W/E/C）以提高可读性
    """
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """
    初始化 Weights & Biases 实验跟踪
    
    参数：
        config: 训练配置
        resuming: 是否恢复之前的训练运行
        log_code: 是否记录代码到 W&B
        enabled: 是否启用 W&B（False 时进入离线模式）
    
    功能：
    - 新训练：创建新的 W&B run 并保存 run ID
    - 恢复训练：从检查点目录读取 run ID 并恢复
    """
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        # 从文件读取之前的 run ID 并恢复
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 创建新的训练运行
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        # 保存 run ID 以便将来恢复
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """
    加载并验证预训练权重
    
    参数：
        loader: 权重加载器（支持从检查点、URL 等加载）
        params_shape: 期望的参数形状（用于验证）
    
    返回值：
        at.Params: 加载的权重参数子集
    
    功能：
    1. 使用加载器加载权重
    2. 验证加载的权重形状和数据类型是否匹配
    3. 移除未加载的参数占位符（jax.ShapeDtypeStruct）
    """
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 从加载的参数中移除 jax.ShapeDtypeStruct（占位符）
    # 这确保只返回实际加载的参数
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """
    初始化训练状态
    
    参数：
        config: 训练配置
        init_rng: 随机数生成器密钥
        mesh: JAX 设备网格（用于分布式训练）
        resume: 是否恢复训练（True 时只返回形状不初始化）
    
    返回值：
        tuple[TrainState, Sharding]: 训练状态和分片规范
    
    训练状态包含：
    - step: 当前训练步数
    - params: 模型参数
    - model_def: 模型定义（GraphDef）
    - tx: 优化器
    - opt_state: 优化器状态
    - ema_params: EMA（指数移动平均）参数（如果启用）
    """
    # 创建优化器（AdamW 等）
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        """内部初始化函数：创建模型并设置训练状态"""
        rng, model_rng = jax.random.split(rng)
        # 初始化模型（及其参数）
        model = config.model.create(model_rng)

        # 将部分预训练参数合并到模型中
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # 如果 partial_params 不是 state 的子集，会产生错误
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # 将冻结的参数转换为 bfloat16（节省内存）
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 获取训练状态的形状（不实际初始化，节省内存）
    train_state_shape = jax.eval_shape(init, init_rng)
    # 创建 FSDP（完全分片数据并行）分片规范
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # 恢复训练时，只返回形状和分片规范
        return train_state_shape, state_sharding

    # 加载预训练权重
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化训练状态并混合预训练参数
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # 捐赠 partial_params 缓冲区（节省内存）
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """
    执行单步训练
    
    参数：
        config: 训练配置
        rng: 随机数生成器
        state: 当前训练状态
        batch: 训练批次（观察 + 动作标签）
    
    返回值：
        tuple[TrainState, dict]: 更新后的训练状态和训练指标
    
    训练步骤：
    1. 前向传播：计算损失
    2. 反向传播：计算梯度
    3. 优化器更新：应用梯度更新参数
    4. EMA 更新：更新指数移动平均参数（如果启用）
    5. 返回训练指标（损失、梯度范数、参数范数）
    """
    # 合并模型定义和参数
    model = nnx.merge(state.model_def, state.params)
    model.train()  # 设置为训练模式（启用 dropout 等）

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        """损失函数：计算模型预测和真实动作之间的损失"""
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # 为当前步骤生成唯一的随机数种子
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # 过滤掉冻结的参数（只对可训练参数计算梯度）
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 应用优化器更新
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 原地更新模型并返回新的完整状态
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # 创建新的训练状态
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    
    # 更新 EMA 参数（如果启用）
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # 过滤出卷积核参数（用于计算参数范数）
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    
    # 返回训练指标
    info = {
        "loss": loss,                                    # 训练损失
        "grad_norm": optax.global_norm(grads),          # 梯度全局范数
        "param_norm": optax.global_norm(kernel_params),  # 参数全局范数
    }
    return new_state, info


def main(config: _config.TrainConfig):
    """
    主训练循环
    
    执行完整的训练流程：
    1. 初始化：日志、设备网格、数据加载器、训练状态
    2. 训练循环：迭代执行训练步骤
    3. 检查点管理：定期保存模型状态
    4. 实验跟踪：记录训练指标到 W&B
    
    参数：
        config: 训练配置对象
    """
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 验证批次大小可被设备数整除（分布式训练要求）
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 设置 JAX 编译缓存目录（加速重复编译）
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 初始化随机数生成器
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 创建设备网格用于分布式训练
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化检查点管理器
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 创建数据加载器
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 记录第一批数据的图像到 W&B（用于检查数据）
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练状态
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)  # 确保初始化完成
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果恢复训练，从检查点加载状态
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # JIT 编译训练步骤（提高性能）
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),  # 捐赠 train_state 以节省内存
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # 在设备网格上下文中执行训练步骤
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        # 定期记录训练指标
        if step % config.log_interval == 0:
            # 堆叠并平均多个步骤的指标
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        
        # 加载下一批数据
        batch = next(data_iter)

        # 定期保存检查点
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
