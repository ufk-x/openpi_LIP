"""PI0 模型实现：基于流匹配的机器人行为策略模型。

PI0 是一个多模态机器人策略模型，结合了视觉-语言理解和流匹配技术：

核心特性：
1. 视觉-语言编码：使用 SigLIP + Gemma 处理图像和自然语言指令
2. 流匹配生成：通过连续时间流匹配生成动作序列
3. 多模态融合：统一处理视觉观察、语言提示和机器人状态
4. 因果注意力：使用精心设计的注意力掩码控制信息流

PI0 vs PI0.5：
- PI0：使用 MLP 融合时间和动作信息，传统的 Transformer 架构
- PI0.5：使用 adaRMS (adaptive Root Mean Square) 动态注入时间信息到归一化层

数据流程：
1. 前缀编码：图像 + 语言提示 -> 视觉语言 tokens
2. 后缀编码：噪声动作 + 时间步 -> 动作 tokens  
3. 联合处理：通过 Transformer 处理完整序列
4. 动作预测：预测速度场用于流匹配去噪

应用场景：
- 机器人操作任务（如 Aloha、DROID）
- 零样本泛化到新任务
- 多模态指令跟随
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """构建注意力掩码（改编自 big_vision），用 `mask_ar` 表达“分块 + 因果”的信息流。

     这套掩码的关键不是“True/False 直接表示因果/非因果”，而是：
     `mask_ar` 定义了一个随序列位置递增的“块编号（block id）”。

     具体规则：
     - 令 `cumsum = cumsum(mask_ar, axis=1)`（对 True/1 做累加）。
     - token i 允许关注 token j 当且仅当：`cumsum[j] <= cumsum[i]`。

     因此：
     - **同一块（cumsum 相同）内是双向注意力**（块内 token 互相可见）。
     - **块与块之间是因果方向**：后面的块可以看前面的块，前面的块看不到后面的块。

     这使得 `mask_ar` 可以非常紧凑地表达多种注意力模式：

     1) 纯因果（标准 GPT）：
         `mask_ar = [1 1 1 1 1 1]` -> `cumsum = [1 2 3 4 5 6]`，从而只能看见自己及之前。

     2) Prefix-LM（前缀双向 + 后缀因果）：
         `mask_ar = [0 0 0 1 1 1]` -> 前 3 个 token 同块双向；后缀 token 之间因果，且后缀可看前缀。

     3) 分块因果（块内双向、块间因果）：
         例如 `mask_ar = [1 0 0 1 0 0 1 0 0]` 表示 3 个块；每个块内双向，但块 2/3 能看块 1。

     在 PI0/PI0.5 中的用途（最重要）：
     - 让 prefix（图像/语言条件）内部双向融合；
     - 同时确保 prefix **不能**关注到 suffix（状态/动作），避免训练/推理时信息泄露；
     - suffix 仍可以看 prefix，以利用条件进行动作去噪/预测。

     Args:
        input_mask: bool[B, N]，True 表示有效 token，False 表示 padding。
        mask_ar: bool[?B, N]，True 表示“从这里开始进入新块”，False 表示“与前一个 token 同块”。

     Returns:
        bool[B, N, N] 注意力掩码；attn_mask[b, i, j] 为 True 表示 token i 可以关注 token j。
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape) # shape [b, n]
    cumsum = jnp.cumsum(mask_ar, axis=1) # shape [b, n]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None] # shape [b, n, n]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None] # shape [b, n, n]
    return jnp.logical_and(attn_mask, valid_mask) # shape [b, n, n]


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """计算标量位置的正弦-余弦位置编码向量。
    
    实现经典的 Transformer 位置编码，但适用于连续值位置（如时间步）。
    使用不同频率的正弦和余弦函数来编码位置信息，使模型能够学习相对位置关系。
    
    编码公式：
    - PE(pos, 2i) = sin(pos / (period_i))
    - PE(pos, 2i+1) = cos(pos / (period_i))
    - period_i = min_period * (max_period / min_period)^(i / (dim//2))
    
    在 PI0 中的应用：
    - 编码流匹配的时间步 t ∈ [0, 1]
    - min_period=4e-3, max_period=4.0 提供对 [0, 1] 范围的敏感性
    
    Args:
        pos: 要编码的标量位置，形状 [batch]
        embedding_dim: 嵌入维度，必须是偶数
        min_period: 最小周期（高频成分）
        max_period: 最大周期（低频成分）
        
    Returns:
        位置嵌入向量，形状 [batch, embedding_dim]
        
    Raises:
        ValueError: 如果 embedding_dim 不是偶数
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    """PI0 多模态机器人策略模型。
    
    PI0 将视觉-语言理解与流匹配技术结合，生成机器人动作序列。
    模型采用混合专家架构，使用不同的 Transformer 模块处理不同模态。
    """
    
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        """初始化 PI0 模型。
        
        构建完整的多模态架构，包括：
        1. 视觉编码器（SigLIP）：处理多摄像头图像输入
        2. 语言-视觉模型（PaliGemma）：处理视觉和语言的联合理解  
        3. 动作专家（Gemma）：专门用于动作序列建模
        4. 投影层：在不同模态之间进行维度转换
        5. 时间处理模块：根据 PI0/PI0.5 选择不同的时间注入方式
        
        Args:
            config: PI0 配置对象，包含模型架构和超参数
            rngs: JAX 随机数生成器，用于参数初始化
        """
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        # 标记是否使用 PI0.5 变体（adaRMS 时间注入）
        self.pi05 = config.pi05
        
        # 获取 Gemma 模型配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # 视觉-语言理解
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # 动作专家
        
        # 构建语言模型（LLM）：混合专家架构
        # TODO: 将 gemma 重写为 NNX 格式。目前使用桥接器。
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],  # 两个专家配置
                embed_dtype=config.dtype,
                adarms=config.pi05,  # 是否启用 adaRMS
            )
        )
        # 延迟初始化：根据 PI0.5 设置决定哪个专家使用 adaRMS
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        
        # 构建视觉编码器（SigLIP）
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 输出维度匹配 PaliGemma
                variant="So400m/14",  # SigLIP-So-400M 模型，14x14 patch size
                pool_type="none",  # 不使用池化，保留所有 patch tokens
                scan=True,  # 使用 scan 优化内存
                dtype_mm=config.dtype,  # 矩阵乘法精度
            )
        )
        # 使用假图像初始化视觉编码器
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        # 组合视觉-语言模型
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # 动作输入投影：将原始动作维度投影到 Transformer 嵌入空间
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        
        # 根据 PI0 变体选择不同的时间处理方式
        if config.pi05:
            # PI0.5：使用 adaRMS 动态注入时间信息
            # 时间 MLP：处理时间嵌入用于 adaRMS 条件化
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # PI0：传统方式，使用 MLP 融合时间和动作信息
            # 状态投影：将机器人状态投影到嵌入空间
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # 动作-时间融合 MLP：拼接动作和时间嵌入后处理
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        
        # 动作输出投影：将 Transformer 隐藏状态投影回动作维度
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # 训练/评估模式标志，由 model.train() 和 model.eval() 自动设置
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """编码观察的前缀部分（图像 + 语言提示）。
        
        这个方法处理模型输入的"条件"部分，即给定的观察信息：
        1. 多摄像头图像：通过 SigLIP 编码为 patch tokens
        2. 语言指令：通过 Gemma 嵌入层编码为 text tokens
        
        前缀使用全连接注意力（非因果），因为这些是给定的输入条件，
        所有 tokens 都可以互相关注以建立上下文理解。
        
        Args:
            obs: 观察对象，包含图像、语言提示等
            
        Returns:
            tokens: 前缀 tokens，形状 [batch, seq_len, embed_dim]
            input_mask: 输入掩码，标记有效 tokens，形状 [batch, seq_len]
            ar_mask: 自回归掩码，前缀部分全为 False（全连接注意力），形状 [seq_len]
        """
        input_mask = []  # 收集每个模态的输入掩码
        ar_mask = []     # 收集注意力模式掩码
        tokens = []      # 收集所有 token 嵌入
        
        # 1. 嵌入图像
        for name in obs.images:
            # 使用 SigLIP 将图像编码为 patch tokens
            # image_tokens: [batch, num_patches, embed_dim]
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            
            # 将图像掩码从 [batch] 扩展到 [batch, num_patches]
            # 如果图像有效，所有 patch tokens 都有效
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],  # num_patches
                )
            )
            
            # 图像 tokens 之间可以全连接注意（ar_mask = False）
            # 这允许不同 patches 之间交互，建立空间理解
            ar_mask += [False] * image_tokens.shape[1]

        # 2. 添加语言输入（分词后的提示）
        if obs.tokenized_prompt is not None:
            # 使用 Gemma 嵌入层将 token IDs 转换为嵌入向量
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            
            # 图像和语言输入之间全连接注意
            # 这允许视觉和语言信息充分融合
            ar_mask += [False] * tokenized_inputs.shape[1]
            
        # 3. 拼接所有前缀 tokens
        tokens = jnp.concatenate(tokens, axis=1)        # [batch, total_prefix_len, embed_dim]
        input_mask = jnp.concatenate(input_mask, axis=1) # [batch, total_prefix_len]
        ar_mask = jnp.array(ar_mask)                     # [total_prefix_len]
        
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """编码观察的后缀部分（状态 + 噪声动作 + 时间）。
        
        这个方法处理模型需要生成/去噪的部分：
        1. 机器人状态：当前关节角度、夹爪位置等（仅 PI0）
        2. 噪声动作序列：需要通过流匹配去噪的动作
        3. 时间步：流匹配过程中的去噪时间
        
        后缀使用因果注意力，确保动作预测不会看到未来信息。
        
        PI0 vs PI0.5 的区别：
        - PI0：使用状态 token + MLP 融合时间和动作
        - PI0.5：省略状态 token，使用 adaRMS 动态注入时间信息
        
        Args:
            obs: 观察对象，包含机器人状态
            noisy_actions: 加噪的动作序列，形状 [batch, action_horizon, action_dim]
            timestep: 流匹配时间步，形状 [batch]，范围 [0, 1]
            
        Returns:
            tokens: 后缀 tokens，形状 [batch, seq_len, embed_dim]
            input_mask: 输入掩码，形状 [batch, seq_len]
            ar_mask: 自回归掩码，形状 [seq_len]，控制因果注意力
            adarms_cond: adaRMS 条件化向量（仅 PI0.5），形状 [batch, embed_dim] 或 None
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        # PI0：添加状态 token（PI0.5 跳过此步骤）
        if not self.pi05:
            # 将机器人状态投影为单个 token
            # state: [batch, action_dim] -> [batch, 1, embed_dim]
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            
            # 状态 token 使用因果注意力：图像/语言输入不能关注状态或动作
            # 这确保了生成过程中的信息流控制
            ar_mask += [True]

        # 将噪声动作投影到嵌入空间
        # noisy_actions: [batch, action_horizon, action_dim] -> [batch, action_horizon, embed_dim]
        action_tokens = self.action_in_proj(noisy_actions)
        
        # 使用正弦-余弦位置编码嵌入时间步
        # 对 [0, 1] 范围提供敏感性，适合流匹配的时间参数
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0) # [batch, embed_dim]
        
        if self.pi05:
            # PI0.5：使用 adaRMS 动态注入时间信息
            # 时间 MLP：处理时间嵌入用于条件化归一化层
            time_emb = self.time_mlp_in(time_emb)    # [batch, embed_dim]
            time_emb = nnx.swish(time_emb)           # 激活函数
            time_emb = self.time_mlp_out(time_emb)   # [batch, embed_dim]
            time_emb = nnx.swish(time_emb)
            
            # 动作 tokens 直接使用，时间信息通过 adaRMS 注入
            action_expert_tokens = action_tokens
            adarms_cond = time_emb  # 用于 adaRMS 的条件化向量
        else:
            # PI0：使用 MLP 显式融合时间和动作信息（无 adaRMS）
            # 将时间嵌入扩展到所有动作步
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon) # [batch, action_horizon, embed_dim]
            
            # 拼接动作和时间嵌入：[batch, action_horizon, 2*embed_dim]
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            
            # 通过 MLP 融合动作和时间信息
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            
            action_expert_tokens = action_time_tokens
            adarms_cond = None  # PI0 不使用 adaRMS
            
        # 添加处理后的动作 tokens
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))  # [batch, action_horizon]
        
        # 设置动作序列的注意力掩码（分块因果：块内双向、块间因果）
        # 这里的 `ar_mask` 采用“分块因果”语义（见 make_attn_mask 的 docstring）：
        # - `True` 表示“从这里开始一个新块”（block id +1）
        # - `False` 表示“与上一个 token 同块”
        # 因此：
        # - 令第一个动作 token 为 True，可以确保 prefix 看不到动作（防止信息泄露）
        # - 后续动作 token 为 False，表示整段 action_horizon 处在同一块内：动作 token 之间是**双向注意力**
        #   （这更符合 flow matching/扩散式“整段轨迹联合去噪”的建模方式，而非自回归逐步生成）。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        # 拼接所有后缀 tokens
        tokens = jnp.concatenate(tokens, axis=1)        # [batch, suffix_len, embed_dim]
        input_mask = jnp.concatenate(input_mask, axis=1) # [batch, suffix_len]
        ar_mask = jnp.array(ar_mask)                     # [suffix_len]
        
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """计算流匹配训练损失。
        
        实现连续时间流匹配的训练目标：
        1. 随机采样时间步 t ∈ [0, 1]
        2. 在真实动作和噪声之间插值：x_t = t*噪声 + (1-t)*真实动作
        3. 预测速度场：u_t = 噪声 - 真实动作
        4. 最小化预测速度场与真实速度场的 L2 损失
        
        流匹配的优势：
        - 相比扩散模型更直接的训练目标
        - 更快的生成过程（ODE 积分）
        - 更好的数值稳定性
        
        Args:
            rng: JAX 随机数生成器
            observation: 观察数据（图像、状态、语言提示等）
            actions: 真实动作序列，形状 [batch, action_horizon, action_dim]
            train: 是否为训练模式
            
        Returns:
            每个动作步的损失，形状 [batch, action_horizon]
        """
        # 分割随机数生成器用于不同的随机操作
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        
        # 预处理观察数据（数据增强、归一化等）
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 流匹配训练的核心：构造插值路径
        batch_shape = actions.shape[:-2]  # 去掉 action_horizon 和 action_dim
        
        # 1. 采样高斯噪声
        noise = jax.random.normal(noise_rng, actions.shape) # shape [batch, action_horizon, action_dim]
        
        # 2. 采样时间步 t ∈ [0.001, 0.999]
        # 使用 Beta(1.5, 1) 分布，更多采样在接近 1 的区域
        # 避免端点 0 和 1 以提高数值稳定性
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001 # shape [batch]
        
        # 3. 构造插值路径：x_t = t*噪声 + (1-t)*真实动作
        time_expanded = time[..., None, None]  # 广播到动作维度，shape [batch, 1, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        
        # 4. 计算目标速度场：u_t = 噪声 - 真实动作
        # 这是从 x_t 流向真实动作的方向
        u_t = noise - actions # shape [batch, action_horizon, action_dim]

        # 5. 前向传播：一次性处理前缀 + 后缀
        # 编码条件信息（图像、语言）
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # 编码要去噪的内容（状态、噪声动作、时间）
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        
        # 构造完整的注意力掩码
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1) # shape [batch, prefix_len + suffix_len]
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0) # shape [prefix_len + suffix_len]
        attn_mask = make_attn_mask(input_mask, ar_mask) # shape [batch, total_len, total_len]
        
        # 计算位置编码
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        # 通过混合专家 Transformer 处理
        # 前缀使用 PaliGemma 专家，后缀使用动作专家
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], 
            mask=attn_mask, 
            positions=positions, 
            adarms_cond=[None, adarms_cond]  # 只有动作专家使用 adaRMS
        )
        
        # 6. 预测速度场
        # 只使用最后 action_horizon 个 tokens（对应动作序列）
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :]) # shape [batch, action_horizon, action_dim]

        # 7. 计算 L2 损失：||预测速度场 - 真实速度场||²
        # 在动作维度上求均值，返回每个时间步的损失
        return jnp.mean(jnp.square(v_t - u_t), axis=-1) # shape [batch, action_horizon]

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """通过流匹配采样生成动作序列。
        
        实现从噪声到目标分布的ODE积分过程：
        1. 从纯噪声开始（t=1）
        2. 沿着学习到的速度场积分到目标分布（t=0）
        3. 使用欧拉方法进行数值积分
        
        相比扩散模型的优势：
        - 确定性采样过程（给定噪声）
        - 更少的采样步数（通常10-50步 vs 1000步）
        - 更好的采样质量和速度
        
        优化技术：
        - KV 缓存：前缀部分只计算一次，重复使用
        - 增量解码：每步只处理后缀部分
        
        Args:
            rng: JAX 随机数生成器
            observation: 观察数据，作为生成条件
            num_steps: ODE 积分步数，更多步数通常有更好质量
            noise: 初始噪声，如果为 None 则随机采样
        Returns:
            生成的动作序列，形状 [batch, action_horizon, action_dim]
        """
        # 预处理观察数据（推理模式，无数据增强）
        observation = _model.preprocess_observation(None, observation, train=False)
        
        # 注意：我们使用扩散文献中更常见的约定，其中 t=1 是噪声，t=0 是目标分布
        # 是的，这与 PI0 论文相反，抱歉造成困扰
        dt = -1.0 / num_steps  # 时间步长（负数，从 1 到 0）
        batch_size = observation.state.shape[0]
        
        # 初始化噪声（如果未提供）
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 优化：首先用前缀的前向传播填充 KV 缓存
        # 这样前缀部分只需要计算一次，后续步骤可以重复使用
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # 只处理前缀，构建 KV 缓存
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions) # prefix_out is not used, kv_cache is for prefix, shape [batch, prefix_len, embed_dim]

        def step(carry):
            """ODE 积分的单步更新。
            
            Args:
                carry: (当前状态 x_t, 当前时间 t)
                
            Returns:
                (下一状态 x_{t+dt}, 下一时间 t+dt)
            """
            x_t, time = carry
            
            # 编码当前状态和时间
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            # 构造注意力掩码
            # suffix_attn_mask: 后缀 tokens 之间的注意力 (b, suffix_len, suffix_len)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            
            # prefix_attn_mask: 后缀 tokens 对前缀 tokens 的注意力 (b, suffix_len, prefix_len)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            
            # full_attn_mask: 后缀 tokens 对完整序列的注意力 (b, suffix_len, prefix_len + suffix_len)
            # 这控制查询（后缀）如何关注键值（前缀+后缀）
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1) # shape (b, suffix_len, prefix_len + suffix_len)
            
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            
            # 计算后缀 tokens 的位置编码
            # jnp.sum(prefix_mask, axis=-1)[:, None] => shape (b, 1)
            # jnp.cumsum(suffix_mask, axis=-1) => shape (b, suffix_len)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1 # shape (b, suffix_len)

            # 增量前向传播：只处理后缀，重用前缀的 KV 缓存
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],  # 前缀为 None，使用缓存
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,     # 重用前缀的 KV 缓存
                adarms_cond=[None, adarms_cond],
            )
            
            assert prefix_out is None  # 确认前缀输出为空（使用缓存）
            
            # 预测当前时间步的速度场
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # 欧拉方法积分：x_{t+dt} = x_t + dt * v_t
            return x_t + dt * v_t, time + dt

        def cond(carry):
            """循环终止条件：时间是否到达 0。"""
            x_t, time = carry
            # 对浮点误差保持鲁棒性
            return time >= -dt / 2

        # ODE 积分：从 t=1（纯噪声）到 t=0（目标分布）
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
