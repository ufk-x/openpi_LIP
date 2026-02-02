import dataclasses  # 数据类支持（用于配置类）
import logging  # 日志记录
from typing import Any  # 通用类型注解

import einops  # 张量重排/重复工具
import flax.nnx as nnx  # Flax NNX 模块系统
import flax.nnx.bridge as nnx_bridge  # NNX 与旧模块桥接
import jax  # JAX 主库
import jax.numpy as jnp  # JAX NumPy
from typing_extensions import override  # 明确标注覆盖父类方法

from openpi.models import model as _model  # 共享模型基类与类型
import openpi.models.gemma_fast as _gemma  # FAST 版本的 Gemma 模型
import openpi.models.siglip as _siglip  # SigLIP 视觉编码器
from openpi.shared import array_typing as at  # 统一数组类型注解
import openpi.shared.nnx_utils as nnx_utils  # NNX 相关工具

logger = logging.getLogger("openpi")  # 统一日志名

PALIGEMMA_EOS_TOKEN = 1  # PaliGemma 词表中的 EOS token id


def make_attn_mask(input_mask, mask_ar):
    """构建注意力掩码（源自 big_vision 的分块因果写法）。

        这里的 `mask_ar` 不是简单的“True=因果 / False=非因果”，而是用来定义**分块（block）**：

        - 令 `cumsum = cumsum(mask_ar, axis=1)`（对 True/1 做累加）
        - token i 能关注 token j 的条件是：`cumsum[j] <= cumsum[i]`

        推论：
        - **同一块内（cumsum 相同）是双向注意力**（块内 token 互相可见）
        - **块间是因果方向**：后面的块可以看前面的块，前面的块看不到后面的块

        常见模式举例：

        1) 纯因果（标准自回归）：
             `mask_ar = [1 1 1 1 1 1]` -> token i 只能看见 i 及之前。

        2) Prefix-LM（前缀双向 + 后缀因果）：
             `mask_ar = [0 0 0 1 1 1]` -> 前 3 个 token 同块双向；后缀 token 之间因果。

        3) 分块因果（块内双向、块间因果）：
             `mask_ar = [1 0 0 1 0 0 1 0 0]` -> 3 个块；块内双向，后块可看前块。

        在 PI0-FAST 中的用途：
        - 图像 patch tokens：设置为 0（同块），允许图像 token 内部双向融合。
        - 文本/状态/动作（FAST token 序列）：由 tokenizer 产出的 `token_ar_mask` 控制。
            一般是：前缀（Task/State/"Action:"）双向，后缀（动作 token）自回归。

        Args:
            input_mask: bool[B, N]，True 表示有效 token，False 表示 padding。
            mask_ar: int/bool[?B, N]，用于定义分块；True/1 表示开启新块。

        Returns:
            bool[B, N, N]，attn_mask[b, i, j] 为 True 表示 token i 可关注 token j。
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  # 广播到 shape:[b,s]
    cumsum = jnp.cumsum(mask_ar, axis=1)  # block id，shape:[b,s]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # 注意力可见性，shape:[b,s,s]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # padding 掩码，shape:[b,s,s]
    return jnp.logical_and(attn_mask, valid_mask)  # 最终掩码，shape:[b,s,s]


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2  # x shape:[s,emb]
    assert input_mask.ndim == 1  # mask shape:[s]
    assert attn_mask.ndim == 2  # attn shape:[s,s]
    assert x.shape[0] == input_mask.shape[0]  # 序列长度一致
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape  # 方阵
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1  # 有效长度 shape:[]
    x = jnp.roll(x, -seqlen, axis=0)  # 右对齐后 x shape:[s,emb]
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)  # 右对齐后 mask shape:[s]
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))  # 右对齐后 attn shape:[s,s]
    return x, input_mask, attn_mask  # 返回对齐后的结果


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)  # 维度对齐
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)  # one-hot shape:[...,n]
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)  # mask shape:[...,n]
    put_values = jnp.einsum("...i,...in->...n", values, onehot)  # values shape:[...,n]
    return jnp.where(put_mask, put_values, arr)  # 仅在指定位置写入


@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"  # 计算/缓存 dtype
    paligemma_variant: _gemma.Variant = "gemma_2b"  # PaliGemma 变体

    # Set the model specific defaults.
    action_dim: int = 32  # 动作维度
    action_horizon: int = 32  # 动作序列长度
    max_token_len: int = 250  # token 序列最大长度

    # Tokenizer for the fast model.
    fast_model_tokenizer: Any | None = None  # FAST tokenizer 实例
    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None  # tokenizer 配置参数

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST  # 标记为 PI0_FAST 类型

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        return Pi0FAST(self, rngs=nnx.Rngs(rng))  # 构建模型实例

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)  # 图像规格 shape:[b,h,w,3]
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)  # 图像 mask 规格 shape:[b]

        with at.disable_typechecking():  # 关闭类型检查以允许 ShapeDtypeStruct
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,  # 视角1
                    "base_1_rgb": image_spec,  # 视角2
                    "wrist_0_rgb": image_spec,  # 腕部视角
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,  # 视角1 mask
                    "base_1_rgb": image_mask_spec,  # 视角2 mask
                    "wrist_0_rgb": image_mask_spec,  # 腕部视角 mask
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),  # 状态向量 shape:[b,adim]
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),  # token 序列 shape:[b,s]
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),  # token mask shape:[b,s]
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),  # ar mask shape:[b,s]
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),  # loss mask shape:[b,s]
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)  # 动作规格 shape:[b,ah,adim]

        return observation_spec, action_spec  # 返回输入规格

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:  # 若使用 LoRA 变体
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))  # 冻结除 LoRA 外的 LLM
        return nnx.Nothing  # 默认不冻结


class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)  # 初始化 BaseModel
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # 读取 PaliGemma 配置
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,  # 词嵌入 dtype
                cache_dtype=config.dtype,  # KV cache dtype
            )
        )
        llm.lazy_init(rngs=rngs, method="init")  # 延迟初始化 LLM
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 与 LLM 宽度对齐
                variant="So400m/14",  # SigLIP 变体
                pool_type="none",  # 不做全局池化
                scan=True,  # 使用 scan 以节省内存
                dtype_mm=config.dtype,  # 多模态 dtype
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)  # 延迟初始化视觉塔
        self.PaliGemma = nnx.Dict(llm=llm, img=img)  # 打包成统一模块

    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []  # 每段输入的有效 mask（列表元素 shape:[b,s_i]）
        ar_mask = []  # 每段输入的分块因果 mask（列表元素 shape:[b,s_i]）
        token_embeddings = []  # 每段输入的 token 嵌入（列表元素 shape:[b,s_i,emb]）
        # embed images - 图像编码
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)  # 视觉编码 shape:[b,s_img,emb]

            token_embeddings.append(image_token_embeddings)  # 收集图像 token shape:[b,s_img,emb]
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # 图像 tokens 作为“条件输入”的一部分：同块（AR mask = 0）=> 图像 token 之间双向注意力。
            ar_mask.append(0 * input_mask[-1])  # 图像 token 同块 shape:[b,s_img]

        # add tokenized inputs - 文本/状态/动作编码
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"  # token 序列
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"  # token mask
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"  # ar mask
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)  # 文本嵌入 shape:[b,s_txt,emb]
        token_embeddings.append(tokenized_inputs_embeddings)  # 收集文本 token shape:[b,s_txt,emb]
        input_mask.append(obs.tokenized_prompt_mask)  # 收集文本 mask shape:[b,s_txt]
        ar_mask.append(obs.token_ar_mask)  # 收集文本 ar mask shape:[b,s_txt]

        # return embeddings, input mask, and ar mask
        return (
            jnp.concatenate(token_embeddings, axis=1),  # 拼接后 shape:[b,s,emb]
            jnp.concatenate(input_mask, axis=1),  # 拼接后 shape:[b,s]
            jnp.concatenate(ar_mask, axis=1),  # 拼接后 shape:[b,s]
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """PI0-FAST 的训练损失（预训练/VLM 阶段）：token-level 自回归交叉熵。

                这一阶段遵循论文里的 FAST 方案：把动作 chunk 编码为离散 token，并拼接到语言前缀后面。
                模型训练目标是“预测下一个 token”，只在动作 token 区间计算损失。

                关键点：
                - `TokenizeFASTInputs` 会产出：
                    - `observation.tokenized_prompt`: 包含 Task/State + Action tokens 的完整序列
                    - `observation.token_ar_mask`: 控制 prefix 双向、action token 自回归
                    - `observation.token_loss_mask`: 只在 action token（以及必要的后缀）上为 True
                - 这里的 `actions` 参数在 PI0-FAST 训练中更多是“为了走通数据接口/用于 tokenizer 产 token”，
                    真正的监督信号来自 `observation.tokenized_prompt`。

                Returns:
                        返回形状为 [batch] 的平均负对数似然（代码里保持与 BaseModel 接口兼容，命名为 Actions）。
            """
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )  # 预处理输入

        # Compute inputs: one big forward pass of prefix + suffix at once
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)  # 输入嵌入 shape:[b,s,emb]
        attn_mask = make_attn_mask(input_mask, ar_mask)  # 注意力掩码 shape:[b,s,s]

        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        # 预测下一个 token，因此 targets 从第 1 个开始
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:], # tokenized_prompt数值范围：[0, vocab_size-1]
            self.PaliGemma.llm.module.vocab_size, # 词表大小
        )  # 预测下一个 token，shape:[b,s-1,vocab]

        # Each input predicts *next* token, so we don't input the last token.
        # 每个输入预测下一个 token，因此输入去掉最后一个 token
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )  # 仅前向到 pre_logits，shape:[b,s-1,emb]

        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        # 只解码目标部分以节省内存
        # (decoding矩阵乘法开销大，因为是 seq_len x vocab_size 的密集层)
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )  # 只解码目标部分，shape:[b,s-1,vocab]
        logp = jax.nn.log_softmax(logits, axis=-1)  # log 概率 shape:[b,s-1,vocab]

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required"  # loss mask, 只有动作 token 上为 True，只计算这些位置的 loss
        loss_mask = observation.token_loss_mask[:, 1:]  # 对齐 targets shape:[b,s-1]，只有动作 token 上为 True
        token_pplx = jnp.sum(targets * logp, axis=-1)  # token 级 NLL shape:[b,s-1]
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)  # 平均 loss shape:[b]，分母必须 >=1 避免除0

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        """基于 PaliGemma 的自回归解码生成 FAST token 序列。

        这不是 PI0/PI0.5 里“连续动作去噪”的采样，而是 FAST 预训练阶段的典型 LLM 解码：
        - prefix = 图像 tokens + 文本/状态 tokens（条件输入）
        - 生成部分 = 动作 chunk 对应的离散 token（以及可能的 EOS）

        关键机制：
        1) KV cache（缓存注意力的 K/V）：
           先用一遍前向把 prefix 写入 cache，后续每步只喂 1 个新 token，避免 O(T^2) 的重复计算。
        2) 右对齐（right-align）：
           由于 batch 内有效 prefix 长度不一致，右对齐后可以用同一个固定大小的 cache，
           并用 prefix_start 把左侧 padding 屏蔽掉。
        3) positions（位置 id）：
           decode=True 时需要显式 positions；prefix 用 0..len-1，生成步从 len 开始递增。
        4) attention mask：
           解码步的 query 长度为 1，因此 mask 形状为 shape:[b,1,s+T]，只控制“这一新 token 能看哪些历史”。

        Returns:
            生成的 token id 序列（注意：这里返回的是离散 token，不是连续动作向量），shape:[b,T]
        """
        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )  # 推理时图像预处理

        # 1) 将 observation 编码成 prefix 的 token embedding。
        # 注意：embed_inputs 会把“图像 tokens + tokenized_prompt tokens”拼在一起。
        # prefix_token_embeddings: shape:[b,s,emb]
        # prefix_mask:            shape:[b,s]（True=有效，False=padding）
        # prefix_ar_mask:         shape:[b,s]（分块因果信号，后续 make_attn_mask 会用它构造 shape:[b,s,s]）
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)  # prefix 编码 shape:[b,s,emb]
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)  # prefix 掩码 shape:[b,s,s]

        # 2) 右对齐 prefix。
        # 目的：batch 内每个样本的有效 prefix 长度不同；右对齐后左侧 padding 被统一挪到最左，
        # 便于在 decode 阶段用一个固定长度（prefill_size + max_decoding_steps）的 KV cache。
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )  # 对齐到右侧，shape:[b,s,emb]/[b,s]/[b,s,s]
        prefill_size = prefix_token_embeddings.shape[1]  # prefix 张量的固定长度（含 padding）shape:[]
        prefill_len = jnp.sum(prefix_mask, axis=-1)  # 每个样本有效 prefix 长度 shape:[b]
        prefix_start = prefill_size - prefill_len  # 每个样本有效区间起点（右对齐后）shape:[b]

        # 3) Prefill：用一次前向把 prefix 写入 KV cache。
        # decode=True 表示走“带 KV cache 的解码模式”。
        # 我们需要把 attention mask 的 key 维度 pad 到 s+T，提前为未来生成步预留 cache 槽位。
        prefix_attn_mask = jnp.pad(
            prefix_attn_mask,
            ((0, 0), (0, 0), (0, max_decoding_steps)),
        )  # 扩展 KV 长度 shape:[b,s,s+T]

        # prefix_positions: 对有效 token 从 0 开始递增，padding 处会得到 -1（通常在实现里会被 mask 掉）
        # shape:[b,s]
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1  # prefix 位置编码 shape:[b,s]
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )  # 预填充 KV cache，logits shape:[b,s,vocab]

        # 4) 初始化解码。
        # prefix_logits 的最后一个位置对应“prefix 最后一个输入 token”的预测分布，
        # 用它来采样生成的第 1 个 token。
        last_logit = prefix_logits[:, -1:]  # 初始解码 logits shape:[b,1,vocab]

        # output_tokens 存放生成结果（token id）。shape:[b,T]
        # 注意：这里用 jnp.zeros 默认 dtype 可能是 float，属于实现细节（不影响注释目标）。
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))  # 输出 token 容器 shape:[b,T]

        # 5) 单步解码函数：输入当前的 last_logit + cache，采样一个 token，并把它写入 cache。
        # carry:
        # - rng:          PRNGKey
        # - last_logit:   shape:[b,1,vocab]（上一步的预测分布）
        # - output_tokens shape:[b,T]
        # - cache:        KV cache（内部结构由 PaliGemma 实现决定）
        # - all_eos:      shape:[] bool（是否全部样本停止）
        # - step:         当前步（从 0 开始）shape:[]
        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry  # 解包状态

            # 5.1) 从 last_logit 采样 token。
            # temperature=0 时使用 argmax（贪心）；temperature>0 时做随机采样。
            rng, rng_step = jax.random.split(rng)  # 为本步采样分 RNG
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )  # 采样或贪心，token shape:[b,1]

            # 5.2) 写入输出缓存。
            # put_along_last_axis 会把 output_tokens[:, step] 更新为 token。
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)  # 写入 token shape:[b,T]

            # 5.3) 早停（EOS）：当所有 batch 样本“本步生成的 token”是 EOS 时停止。
            # 注意：这里没有累积“历史是否已出现 EOS”，因此只有“同一步全体 EOS”才会停。
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)  # 本步是否生成 EOS shape:[b]
            all_eos = jnp.all(has_eos)  # 是否全部结束 shape:[]

            # 5.4) 单步前向：把 token 转 embedding，并用 KV cache 增量解码。
            # token_embedding: shape:[b,1,emb]
            token_embedding = self.PaliGemma.llm(token, embed_only=True)  # token 嵌入 shape:[b,1,emb]

            # positions: 当前生成 token 的位置 id。
            # prefix 的有效长度是 prefill_len，因此第一个生成 token 的位置是 prefill_len（再 +1 对齐实现细节）。
            positions = prefill_len[:, None] + step + 1  # 当前位置 shape:[b,1]

            # mask: shape:[b,1,s+T]
            # 目标：当前新 token 只能看见（1）有效 prefix 区间（剔除左侧 padding），以及（2）已经生成的历史 token。
            # 下界：>= prefix_start，屏蔽掉右对齐后的左侧 padding。
            # 上界：< prefill_size + step + 1，允许看到 prefix（长度 prefill_size）以及已生成的 step+1 个 token。
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )  # 解码阶段注意力 mask shape:[b,1,s+T]
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )  # 单步解码，logits shape:[b,1,vocab]

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1  # 更新状态

        def cond(carry):
            _, _, _, _, all_eos, step = carry  # 解包状态
            return (~all_eos) & (step < max_decoding_steps)  # 未结束且未超长

        # 6) 用 lax.while_loop 包住解码循环，便于 JIT 编译整个生成过程。
        # 退出条件：全部 EOS 或达到 max_decoding_steps。
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )  # 运行解码循环
        return output_tokens  # 返回生成的 token 序列 shape:[b,T]
