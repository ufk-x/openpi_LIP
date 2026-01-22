"""PI0 (PyTorch) 模型实现。

PI0 是一个多模态机器人策略模型，结合了视觉、语言和机器人状态信息来生成动作序列。
该实现基于以下核心组件：

1. **多模态输入处理**：
   - 图像：通过 SigLIP 视觉编码器处理
   - 语言：通过 PaliGemma 的语言编码器处理
   - 机器人状态：通过线性层投影

2. **扩散/流匹配框架**：
   - 训练时：学习从噪声+动作混合状态预测速度场
   - 推理时：通过 Euler 方法从纯噪声迭代生成最终动作

3. **注意力机制设计**：
   - Prefix tokens (图像+语言) 可以相互关注
   - Suffix tokens (状态+动作) 采用因果注意力
   - Prefix 和 Suffix 之间的注意力受控制

4. **两种架构模式**：
   - PI0.5: 使用 adaRMS 归一化，时间信息通过条件向量传递
   - 标准 PI0: 时间和动作信息通过 MLP 融合

主要数据流：
观测输入 -> 预处理 -> embed_prefix (图像+语言) + embed_suffix (状态+动作+时间) 
-> 组合注意力掩码 -> Transformer 前向 -> 动作输出
"""

import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """根据设备类型返回安全的数据类型。
    
    某些设备（如 CPU）可能不支持特定的数据类型（如 bfloat16），
    此函数会自动回退到兼容的数据类型以避免运行时错误。
    
    Args:
        target_dtype: 目标数据类型
        device_type: 设备类型字符串（如 'cpu', 'cuda'）
        
    Returns:
        torch.dtype: 在指定设备上安全可用的数据类型
    """
    if device_type == "cpu":
        # CPU 不支持 bfloat16，回退到 float32
        if target_dtype == torch.bfloat16:
            return torch.float32
        # 保持 float64 以维持精度
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """为标量时间位置计算正弦-余弦位置编码。
    
    这是 Transformer 中常用的位置编码方法的变体，用于将连续的时间值
    （在扩散模型中通常是 [0, 1] 范围内的时间步）编码为高维向量。
    不同频率的正弦余弦函数能够捕捉不同时间尺度的模式。
    
    Args:
        time: 形状为 [batch_size] 的时间张量
        dimension: 输出嵌入的维度（必须为偶数，前半部分用 sin，后半部分用 cos）
        min_period: 最小周期，对应最高频率
        max_period: 最大周期，对应最低频率  
        device: 计算设备
        
    Returns:
        形状为 [batch_size, dimension] 的位置编码张量
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    # 使用高精度计算避免数值误差
    dtype = get_safe_dtype(torch.float64, device.type)
    # 创建频率分布：从高频到低频呈几何级数变化
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # 计算角频率并与时间相乘
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]  # 广播：[B, 1] * [1, D//2] -> [B, D//2]
    # 拼接正弦和余弦部分
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    """从 Beta 分布中采样时间步。
    
    Beta 分布被用来采样扩散过程中的时间步 t，不同的 alpha/beta 参数
    会产生不同的时间分布偏好（如偏向噪声端或干净端）。
    
    Args:
        alpha: Beta 分布的 alpha 参数
        beta: Beta 分布的 beta 参数  
        bsize: 批次大小
        device: 计算设备
        
    Returns:
        形状为 [bsize] 的采样结果
    """
    # 确保参数在正确的设备和数据类型上
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """构造 2D 注意力掩码以控制 token 之间的注意力模式。
    
    此函数来自 big_vision，用于实现灵活的注意力模式：
    - 因果注意力（causal attention）
    - 前缀-语言模型注意力（prefix-LM attention） 
    - 分块注意力（block attention）
    
    核心思想：token i 可以关注 token j 当且仅当 segment_id(j) <= segment_id(i)
    
    注意力模式示例：
      [1 1 1 1 1 1]: 纯因果注意力（每个 token 只能看到之前的 token）
      [0 0 0 1 1 1]: 前缀-LM 注意力（前 3 个 token 互相可见，后 3 个采用因果）
      [1 0 1 0 1 0 0 1 0 0]: 4 个块之间的注意力（块内全连接，只能看到之前的块）

    Args:
        pad_masks: bool[B, N] 填充掩码，True 表示有效 token
        att_masks: int[B, N] 注意力控制掩码，1 表示开始新的注意力段
        
    Returns:
        bool[B, N, N]: 2D 注意力掩码，True 表示位置 (i,j) 可以相互关注
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # 通过累积和计算每个位置的段 ID
    cumsum = torch.cumsum(att_masks, dim=1)
    # 段规则：token i 可以关注 token j 当且仅当 segment_id(j) <= segment_id(i)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N, N]
    # 填充掩码：只有两个位置都是有效 token 时才能相互关注
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]  # [B, N, N]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    """PI0 多模态机器人策略模型的 PyTorch 实现。
    
    该模型将视觉、语言和机器人状态信息融合，使用扩散/流匹配框架
    生成机器人动作序列。模型由两个主要部分组成：
    1. PaliGemma: 处理视觉和语言输入
    2. Expert Gemma: 处理机器人状态和动作生成
    """
    
    def __init__(self, config):
        """初始化 PI0 模型。
        
        Args:
            config: 包含模型配置的对象，应包含以下属性：
                - pi05: bool, 是否使用 PI0.5 架构
                - paligemma_variant: PaliGemma 配置名称
                - action_expert_variant: 动作专家模型配置名称
                - dtype: 模型精度
                - action_horizon: 动作序列长度
                - action_dim: 动作维度（通常为 32）
        """
        super().__init__()
        self.config = config
        self.pi05 = config.pi05  # 控制使用 PI0.5 还是标准 PI0 架构

        # 获取子模型配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # 创建组合模型：PaliGemma (视觉+语言) + Expert Gemma (动作)
        # use_adarms: [PaliGemma, Expert] 是否使用 adaRMS 归一化
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],  # PI0.5 在 Expert 中使用 adaRMS
            precision=config.dtype,
        )

        # 动作输入输出投影层（机器人动作固定为 32 维）
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            # PI0.5 架构：时间信息通过 adaRMS 条件传递，需要时间 MLP
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # 标准 PI0 架构：显式处理状态，时间和动作通过 MLP 融合
            self.state_proj = nn.Linear(32, action_expert_config.width)  # 状态投影层
            # 动作+时间融合 MLP（输入维度为两倍 width）
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # 性能优化设置
        torch.set_float32_matmul_precision("high")  # 使用 TensorFloat-32 加速矩阵乘法
        # 编译推理函数以获得更好的性能（JIT 编译）
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # 梯度检查点标志（用于减少显存占用）
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """开启梯度检查点以优化显存使用。
        
        梯度检查点是一种优化技巧，通过在前向传播时不存储部分中间激活，
        而在反向传播时重新计算来节省显存。这对于训练大模型非常有用。
        """
        self.gradient_checkpointing_enabled = True
        # 为各个子模块开启梯度检查点
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点。
        
        禁用后模型将恢复正常的前向传播行为（存储所有中间激活）。
        """
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """检查梯度检查点是否开启。
        
        Returns:
            bool: 如果开启了梯度检查点则返回 True
        """
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """根据配置有条件地应用梯度检查点。
        
        如果开启了梯度检查点且模型处于训练模式，则使用梯度检查点执行函数；
        否则直接执行函数。
        
        Args:
            func: 要执行的函数
            *args: 函数位置参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        if self.gradient_checkpointing_enabled and self.training:
            # 使用梯度检查点：
            # - use_reentrant=False: 使用更稳定的非重入实现
            # - preserve_rng_state=False: 不保存随机数状态以节省内存
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """将 2D 注意力掩码转换为 Transformer 所需的 4D 格式。
        
        Transformer 模型期望的是加法注意力掩码（additive attention mask）：
        - 允许注意力的位置加 0
        - 禁止注意力的位置加一个很大的负数（约等于 -∞）
        
        Args:
            att_2d_masks: bool[B, N, N] 2D 注意力掩码
            
        Returns:
            float[B, 1, N, N]: 4D 加法注意力掩码
        """
        # 添加维度以匹配 Transformer 的期望输入格式 [B, num_heads, N, N]
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # 转换为加法掩码：True -> 0，False -> -∞
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """预处理观测数据并解包为各个组件。
        
        将上层传入的 observation 对象转换为模型可以直接使用的张量格式。
        
        Args:
            observation: 原始观测对象
            train: 是否为训练模式（影响数据增强等）
            
        Returns:
            tuple: (images, image_masks, tokenized_prompt, tokenized_prompt_mask, state)
                - images: 图像列表
                - image_masks: 图像掩码列表
                - tokenized_prompt: 词元化后的提示文本
                - tokenized_prompt_mask: 提示文本掩码
                - state: 机器人状态向量
        """
        # 调用预处理模块进行数据转换
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),      # 多视角图像
            list(observation.image_masks.values()), # 图像有效区域掩码
            observation.tokenized_prompt,           # 指令文本的 token 序列
            observation.tokenized_prompt_mask,      # 指令文本的有效 token 掩码
            observation.state,                      # 机器人状态向量（关节角度、末端位置等）
        )

    def sample_noise(self, shape, device):
        """采样标准高斯噪声。
        
        用于扩散模型的噪声采样，从正态分布 N(0, I) 中采样。
        
        Args:
            shape: 噪声张量的形状
            device: 计算设备
            
        Returns:
            torch.Tensor: 指定形状的高斯噪声张量
        """
        return torch.normal(
            mean=0.0,           # 均值为 0
            std=1.0,            # 标准差为 1
            size=shape,
            dtype=torch.float32, # 使用 float32 保证精度
            device=device,
        )

    def sample_time(self, bsize, device):
        """采样扩散时间步。
        
        使用 Beta 分布采样时间步 t，然后调整到 [0.001, 1.0) 范围。
        Beta(1.5, 1.0) 会偏向采样较大的 t 值（靠近纯噪声端）。
        
        Args:
            bsize: 批次大小
            device: 计算设备
            
        Returns:
            torch.Tensor: 形状为 [bsize] 的时间步张量
        """
        # 从 Beta(1.5, 1.0) 采样，偏向较大的值
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        # 缩放到 [0.001, 1.0) 范围，避免 t=0 时的数值问题
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码前缀部分：图像和语言输入。
        
        这个方法处理模型的视觉和语言输入，将它们编码为统一的嵌入表示。
        这些“前缀 tokens”将与后续的动作 tokens 一起输入 Transformer。
        
        数据流：
        1. 图像通过 SigLIP 视觉编码器处理
        2. 语言 tokens 通过嵌入层处理并乘以 sqrt(d) 缩放
        3. 组合填充掩码和注意力掩码
        
        Args:
            images: 多视角图像列表，每个元素形状为 [B, H, W, C]
            img_masks: 对应的图像掩码列表，标记有效图像区域
            lang_tokens: 词元化的语言指令，形状为 [B, L]
            lang_masks: 语言 tokens 的有效性掩码，形状为 [B, L]
            
        Returns:
            tuple: (embeddings, padding_masks, attention_masks)
                - embeddings: 合并后的嵌入表示 [B, N_prefix, D]
                - padding_masks: 填充掩码 [B, N_prefix]，True 表示有效 token
                - attention_masks: 注意力掩码 [B, N_prefix]，控制注意力模式
        """
        embs = []           # 存储各部分的嵌入表示
        pad_masks = []      # 存储填充掩码
        att_masks = []      # 存储注意力掩码

        # 处理多视角图像
        for img, img_mask in zip(images, img_masks, strict=True):
            # 使用梯度检查点优化图像编码过程
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]  # [B, N_img, D]

            embs.append(img_emb)
            # 扩展图像掩码以匹配图像嵌入的 token 数量
            # img_mask: [B, ] -> [B, 1] ->[B, N_img]
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # 图像 tokens 之间采用全连接注意力（0 表示同一注意力段）
            att_masks += [0] * num_img_embs

        # 处理语言 tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            # 乘以 sqrt(d) 进行尺度缩放（Transformer 常用做法）
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens) # shape: [B, L, D]
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # 图像和语言之间也采用全连接注意力
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # 合并所有嵌入和掩码
        embs = torch.cat(embs, dim=1)           # [B, N_img + N_lang, D]
        pad_masks = torch.cat(pad_masks, dim=1) # [B, N_img + N_lang]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # 将 1D 注意力掩码扩展为批次维度
        bsize = pad_masks.shape[0]
        # att_masks: [N_prefix] -> [1, N_prefix] -> [B, N_prefix]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))  # [B, N_prefix]

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """编码后缀部分：机器人状态、噪声动作和时间步。
        
        这个方法处理模型的动作相关输入，包括机器人当前状态、
        扩散过程中的噪声动作 x_t 和时间步 t。这些“后缀 tokens”
        将与前缀 tokens 一起输入 Expert Gemma 模型。
        
        两种架构模式：
        - 标准 PI0: 显式添加状态 token，时间和动作通过 MLP 融合
        - PI0.5: 时间信息通过 adaRMS 条件传递，不显式加入状态 token
        
        Args:
            state: 机器人状态向量 [B, state_dim]，通常为 32 维
            noisy_actions: 噪声动作 x_t [B, action_horizon, action_dim]
            timestep: 扩散时间步 [B]，取值范围 [0, 1]
            
        Returns:
            tuple: (embeddings, padding_masks, attention_masks, adarms_cond)
                - embeddings: 后缀嵌入表示 [B, N_suffix, D]
                - padding_masks: 填充掩码 [B, N_suffix]
                - attention_masks: 注意力掩码 [B, N_suffix]
                - adarms_cond: adaRMS 条件向量（仅 PI0.5 模式）
        """
        embs = []           # 存储各部分的嵌入表示
        pad_masks = []      # 存储填充掩码
        att_masks = []      # 存储注意力掩码

        # 标准 PI0 架构：显式处理机器人状态
        if not self.pi05:
            # 确保数据类型一致性
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # 将机器人状态投影到高维空间
            def state_proj_func(state):
                return self.state_proj(state) # nn.Linear(32, action_expert_config.width)，[B, 32]到[B,width]

            state_emb = self._apply_checkpoint(state_proj_func, state)
            embs.append(state_emb[:, None, :])  # 添加序列维度 [B, 1, D]
            
            bsize = state_emb.shape[0]
            device = state_emb.device
            # 为状态 token 创建掩码
            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device) # [B, 1]
            pad_masks.append(state_mask)

            # 注意力掩码：1 表示开始新的注意力段（防止前缀 tokens 关注状态）
            att_masks += [1]

        # 编码时间步：使用正弦-余弦位置编码
        time_emb = create_sinusoidal_pos_embedding(
            timestep, 
            self.action_in_proj.out_features,  # 与动作嵌入维度一致
            min_period=4e-3,    # 最小周期，捕捉高频变化
            max_period=4.0,     # 最大周期，捕捉低频变化
            device=timestep.device
        ) # [B, D]
        time_emb = time_emb.type(dtype=timestep.dtype)  # 保持数据类型一致

        # 编码噪声动作：投影到高维空间
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions) # [B, action_horizon, D]

        if not self.pi05:
            # 标准 PI0: 时间和动作信息通过 MLP 融合
            # 将时间嵌入扩展到每个动作 token
            time_emb = time_emb[:, None, :].expand_as(action_emb)  # [B, action_horizon, D]
            # 拼接动作和时间嵌入
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # [B, H, 2*D]

            # 通过 MLP 融合时间和动作信息
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # SiLU 激活函数（也称为 Swish）
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None  # 标准模式不使用 adaRMS 条件
        else:
            # PI0.5: 时间信息通过 adaRMS 条件传递
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)  # 双重 SiLU 激活

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb  # 动作不与时间直接融合
            adarms_cond = time_emb        # 时间作为 adaRMS 条件

        # 将动作 tokens 添加到嵌入列表
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        # 为所有动作 tokens 创建有效掩码
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # 动作 tokens 的注意力模式：第一个动作 token 开始新段，后续 tokens 在同一段内
        # 这实现了动作 tokens 的因果注意力模式
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        # 合并所有后缀嵌入和掩码
        embs = torch.cat(embs, dim=1)       # [B, N_suffix, D]
        pad_masks = torch.cat(pad_masks, dim=1)  # [B, N_suffix]
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))  # [B, N_suffix]

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """模型训练前向传播：计算流匹配损失。
        
        该方法实现了流匹配（Flow Matching）框架下的训练过程。
        核心思想是学习一个速度场，能够将噪声转换为目标动作。
        
        数学原理：
        - 给定时间 t ∈ [0,1] 和噪声 ε，构造混合状态：x_t = t*ε + (1-t)*x
        - 目标速度场：u_t = ε - x
        - 训练模型预测：v_t ≈ u_t
        
        Args:
            observation: 多模态观测数据，包含图像、指令和机器人状态
            actions: 目标动作序列 [B, action_horizon, action_dim]
            noise: 可选的噪声张量，不提供则随机采样
            time: 可选的时间步，不提供则从 Beta 分布采样
            
        Returns:
            Tensor: 逐元素 MSE 损失 [B, action_horizon, action_dim]（未做 reduction）
        """
        # 1. 预处理多模态观测数据
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        # 2. 采样或使用提供的噪声
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        # 3. 采样或使用提供的时间步
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 4. 构造流匹配目标：混合状态和速度场
        time_expanded = time[:, None, None]  # 扩展到动作张量的形状 [B, 1, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # 混合状态
        u_t = noise - actions  # 目标速度场

        # 5. 编码前缀（图像+语言）和后缀（状态+动作+时间）
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        
        # 6. 数据类型对齐（优化性能）
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 7. 组合前缀和后缀的掩码
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # 8. 构造 2D 注意力掩码和位置 ID
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1  # 位置编号从 0 开始

        # 9. 转换为 Transformer 所需的 4D 注意力掩码
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 10. Transformer 前向传播（可选梯度检查点）
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            # 调用组合模型，只关注后缀部分的输出（动作相关）
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,        # 训练时不使用 KV 缓存
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,             # 训练时不缓存 KV
                adarms_cond=[None, adarms_cond],  # adaRMS 条件（仅用于 Expert）
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ) # shape: [B, N_suffix, D]

        # 11. 提取动作相关的输出
        suffix_out = suffix_out[:, -self.config.action_horizon :]  # 只保留后 action_horizon 个 tokens，shape: [B, H, D]
        suffix_out = suffix_out.to(dtype=torch.float32)  # 转换为 float32 保证精度

        # 12. 最终的动作预测（可选梯度检查点）
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # 13. 计算逐元素 MSE 损失（不做 reduction）
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """模型推理：从噪声生成动作序列。
        
        该方法实现了流匹配框架下的推理过程。通过求解 ODE：
        dx/dt = v(x_t, t)，从 t=1（纯噪声）积分到 t=0（目标动作）。
        
        关键优化：
        - 使用 KV 缓存加速：前缀部分（图像+语言）只算一次
        - 每步只对后缀部分（动作）进行前向传播
        - 使用 Euler 方法求解 ODE
        
        Args:
            device: 计算设备
            observation: 多模态观测数据
            noise: 可选的初始噪声，不提供则随机采样
            num_steps: Euler 积分步数，越大越精确但越慢
            
        Returns:
            Tensor: 生成的动作序列 [B, action_horizon, action_dim]
        """
        bsize = observation.state.shape[0]
        
        # 1. 初始化噪声（如果未提供）
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 2. 预处理观测数据
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # 3. 编码前缀并构建 KV 缓存
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 4. 生成前缀部分的 KV 缓存（仅算一次）
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        # 使用 eager attention 实现（更稳定）
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # 只对前缀做前向传播，获得 KV 缓存
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # 只处理前缀
            use_cache=True,  # 开启 KV 缓存
        )

        # 5. 设置 Euler 积分参数
        dt = -1.0 / num_steps  # 负数：从 t=1 积分到 t=0
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # 6. Euler 积分循环：从噪声迭代到动作
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)  # 从 t=1 开始
        
        while time >= -dt / 2:  # 终止条件：接近 t=0
            expanded_time = time.expand(bsize)  # 扩展到批次维度
            
            # 计算当前时刻的速度场 v_t
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler 更新：x_{t+dt} = x_t + dt * v_t
            x_t = x_t + dt * v_t
            time += dt
            
        return x_t  # 最终的去噪声结果

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """执行单步去噪：预测当前时刻 x_t 的速度场 v_t。
        
        这个方法是推理过程的核心，在给定的时间步 t 下，
        根据当前的噪声状态 x_t 预测速度场。
        
        关键优化：
        - 复用前缀的 KV 缓存，只对后缀部分计算
        - 构造组合注意力掩码（前缀+后缀）
        - 正确处理位置编号的偏移
        
        Args:
            state: 机器人状态向量 [B, state_dim]
            prefix_pad_masks: 前缀填充掩码 [B, N_prefix]
            past_key_values: 前缀部分的 KV 缓存
            x_t: 当前噪声状态 [B, action_horizon, action_dim]
            timestep: 当前时间步 [B]
            
        Returns:
            Tensor: 预测的速度场 v_t [B, action_horizon, action_dim]
        """
        # 1. 编码后缀部分（状态+动作+时间）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # 2. 获取各部分的尺寸信息
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # 3. 构造组合注意力掩码
        # 前缀部分：后缀可以关注所有前缀 tokens（受填充掩码控制）
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        # 后缀部分：自身的因果/分块注意力模式
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        # 拼接成完整的注意力掩码
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # 4. 计算后缀部分的位置 ID（在前缀之后继续编号）
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]  # 前缀的有效 token 数
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # 5. 转换为 4D 注意力掩码
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        # 为 Expert Gemma 设置 eager attention
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # 6. 前向传播（使用 KV 缓存）
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,  # 复用前缀的 KV 缓存
            inputs_embeds=[None, suffix_embs],    # 只处理后缀部分
            use_cache=False,                      # 不需要更新缓存
            adarms_cond=[None, adarms_cond],      # adaRMS 条件仅用于 Expert
        ) # shape: [B, N_suffix, D]，N_suffix的值为 state token 数 + action_horizon

        # 7. 提取动作相关的输出并投影到动作空间
        suffix_out = outputs_embeds[1]  # Expert Gemma 的输出
        suffix_out = suffix_out[:, -self.config.action_horizon :]  # 只保留动作 tokens，同时适配是否有状态 token
        suffix_out = suffix_out.to(dtype=torch.float32)  # 转换为高精度
        
        # 8. 投影到动作空间并返回速度场
        return self.action_out_proj(suffix_out)
