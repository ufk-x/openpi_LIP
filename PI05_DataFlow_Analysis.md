# PI0.5 模型数据流分析：从 Observation 到 Action

本文档详细追踪 PI0.5 模型中从 observation 输入到 action 输出的完整数据流，包括每个阶段的维度变化以及从 VLM（视觉语言模型）到 Flow Matching（流匹配）的转换过程。

## 目录
- [概述](#概述)
- [数据结构定义](#数据结构定义)
- [完整数据流](#完整数据流)
- [VLM 处理阶段](#vlm-处理阶段)
- [Flow Matching 阶段](#flow-matching-阶段)
- [推理过程](#推理过程)

---

## 概述

PI0.5 模型是一个多模态机器人策略模型，它结合了：
- **视觉编码**：通过 SigLIP 处理多视角图像
- **语言理解**：通过 PaliGemma 处理自然语言指令
- **动作生成**：通过 Expert Gemma + Flow Matching 生成动作序列

### PI0.5 vs PI0 的关键区别

来源：[`src/openpi/models/pi0_config.py#L26-L29`](src/openpi/models/pi0_config.py#L26-L29)
```python
# Pi05 has two differences from Pi0:
# - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
# - the action expert uses adaRMSNorm to inject the flow matching timestep
pi05: bool = False
```

**两个主要区别**：
1. **状态输入方式**：PI0.5 将状态作为离散的语言 token 处理，而非连续的后缀输入
2. **时间注入方式**：PI0.5 使用 adaRMSNorm 注入时间信息，而非通过 MLP 融合

---

## 数据结构定义

### Observation 结构

来源：[`src/openpi/models/model.py#L138-L159`](src/openpi/models/model.py#L138-L159)

```python
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """保存观测数据，即模型的输入。
    
    观测数据包含机器人感知到的所有信息：
    - 多摄像头图像：提供视觉信息，用于理解环境和物体
    - 机器人状态：关节角度、夹爪位置等本体感觉信息
    - 语言提示：任务描述，指导机器人执行特定行为
    """

    # 图像数据，float32格式，范围[-1, 1]
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # 图像掩码，与images具有相同的键
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # 低维机器人状态（关节角度、夹爪位置等）
    state: at.Float[ArrayT, "*b s"]

    # 分词后的提示文本
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # 分词提示的掩码（标识有效token）
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
```

### 标准维度约定

来源：[`src/openpi/models/model.py#L84-L97`](src/openpi/models/model.py#L84-L97)

```python
# 模型始终期望的图像键名
IMAGE_KEYS = (
    "base_0_rgb",           # 主摄像头（固定外部视角）
    "left_wrist_0_rgb",     # 左臂腕部摄像头
    "right_wrist_0_rgb",    # 右臂腕部摄像头
)

# 标准图像分辨率
IMAGE_RESOLUTION = (224, 224)
```

### 模型配置

来源：[`src/openpi/models/pi0_config.py#L18-L29`](src/openpi/models/pi0_config.py#L18-L29)

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32              # 动作维度
    action_horizon: int = 50          # 动作序列长度
    max_token_len: int = None         # PI0.5 默认为 200
    pi05: bool = False                # 是否启用 PI0.5 模式
```

**关键维度**：
- `action_dim = 32`：动作空间维度（通常对应机器人的关节数+夹爪）
- `action_horizon = 50`：预测的动作序列长度
- `max_token_len = 200`（PI0.5）：最大 token 序列长度

---

## 完整数据流

### 阶段 1：输入预处理

**输入**：原始 observation 对象

来源：[`src/openpi/models_pytorch/pi0_pytorch.py#L288-L310`](src/openpi/models_pytorch/pi0_pytorch.py#L288-L310)

```python
def _preprocess_observation(self, observation, *, train=True):
    """预处理观测数据并解包为各个组件。"""
    observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
    return (
        list(observation.images.values()),      # 多视角图像
        list(observation.image_masks.values()), # 图像有效区域掩码
        observation.tokenized_prompt,           # 指令文本的 token 序列
        observation.tokenized_prompt_mask,      # 指令文本的有效 token 掩码
        observation.state,                      # 机器人状态向量
    )
```

**维度转换**：
- `images`: 3 × `[B, 224, 224, 3]` → List of 3 tensors
- `image_masks`: 3 × `[B]` → List of 3 tensors
- `tokenized_prompt`: `[B, L]` (L ≤ 200)
- `tokenized_prompt_mask`: `[B, L]`
- `state`: `[B, 32]`

### 阶段 2：Prefix Embedding（VLM 处理）

来源：[`src/openpi/models_pytorch/pi0_pytorch.py#L312-L380`](src/openpi/models_pytorch/pi0_pytorch.py#L312-L380)

#### 2.1 图像编码

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    """编码前缀部分：图像和语言输入。"""
    embs = []
    pad_masks = []
    att_masks = []

    # 处理多视角图像
    for img, img_mask in zip(images, img_masks, strict=True):
        def image_embed_func(img):
            return self.paligemma_with_expert.embed_image(img)

        img_emb = self._apply_checkpoint(image_embed_func, img)
        bsize, num_img_embs = img_emb.shape[:2]  # [B, N_img, D]

        embs.append(img_emb)
        # 扩展图像掩码以匹配图像嵌入的 token 数量
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        # 图像 tokens 之间采用全连接注意力
        att_masks += [0] * num_img_embs
```

**图像编码维度变化**：
- 输入：`[B, 224, 224, 3]`
- 经过 SigLIP Vision Encoder：`[B, 224, 224, 3]` → `[B, (224/14)², D]` = `[B, 256, D]`
  - Patch size = 14x14
  - Number of patches = (224/14)² = 16² = 256
  - D = PaliGemma hidden size (通常为 2048)

**3 个视角图像**：
- 总共：3 × 256 = 768 个图像 tokens
- 维度：`[B, 768, D]`

#### 2.2 语言编码

```python
    # 处理语言 tokens
    def lang_embed_func(lang_tokens):
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        # 乘以 sqrt(d) 进行尺度缩放
        return lang_emb * math.sqrt(lang_emb_dim)

    lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
    embs.append(lang_emb)
    pad_masks.append(lang_masks)
```

**语言编码维度变化**：
- 输入：`[B, L]` (token IDs)
- 经过 Embedding Layer：`[B, L]` → `[B, L, D]`
- 缩放后：`[B, L, D] * sqrt(D)`

#### 2.3 Prefix 组合

```python
    # 合并所有嵌入和掩码
    embs = torch.cat(embs, dim=1)           # [B, N_img + N_lang, D]
    pad_masks = torch.cat(pad_masks, dim=1) # [B, N_img + N_lang]
    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
```

**Prefix 最终维度**：
- `prefix_embs`: `[B, N_prefix, D]` 其中 `N_prefix = 768 + L`
- `prefix_pad_masks`: `[B, N_prefix]`
- `prefix_att_masks`: `[B, N_prefix]` (全 0 = 全连接注意力)

### 阶段 3：Suffix Embedding（Flow Matching 准备）

来源：[`src/openpi/models_pytorch/pi0_pytorch.py#L382-L550`](src/openpi/models_pytorch/pi0_pytorch.py#L382-L550)

#### 3.1 时间步编码

```python
def embed_suffix(self, state, noisy_actions, timestep):
    """编码后缀部分：机器人状态、噪声动作和时间步。"""
    
    # 编码时间步：使用正弦-余弦位置编码
    time_emb = create_sinusoidal_pos_embedding(
        timestep, 
        self.action_in_proj.out_features,  # 与动作嵌入维度一致
        min_period=4e-3,
        max_period=4.0,
        device=timestep.device
    )
```

**时间编码维度变化**：
- 输入：`timestep [B]` (范围 [0.001, 1.0])
- 输出：`time_emb [B, D_expert]` (D_expert = Expert Gemma width，通常 1024)

#### 3.2 动作编码

```python
    # 编码噪声动作：投影到高维空间
    def action_proj_func(noisy_actions):
        return self.action_in_proj(noisy_actions)

    action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
```

**动作编码维度变化**：
- 输入：`noisy_actions [B, 50, 32]`
- 经过 Linear(32, D_expert)：`[B, 50, 32]` → `[B, 50, D_expert]`

#### 3.3 PI0.5 特殊处理：adaRMS 条件

```python
    if self.pi05:
        # PI0.5: 时间信息通过 adaRMS 条件传递
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb  # 动作不与时间直接融合
        adarms_cond = time_emb        # 时间作为 adaRMS 条件
```

**PI0.5 时间处理**：
- `time_emb [B, D_expert]` → MLP → `adarms_cond [B, D_expert]`
- 动作 tokens 保持不变：`action_time_emb = action_emb [B, 50, D_expert]`
- **关键**：时间信息不直接加入 tokens，而是通过 adaRMS 注入 Transformer 层

对比 **标准 PI0 的处理**：
```python
    if not self.pi05:
        # 标准 PI0: 时间和动作信息通过 MLP 融合
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # [B, H, 2*D]
        
        # 通过 MLP 融合
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        adarms_cond = None
```

#### 3.4 状态处理差异

**PI0（标准版）**：
```python
    if not self.pi05:
        # 标准 PI0：显式处理机器人状态
        state_emb = self.state_proj(state)  # [B, 32] → [B, D_expert]
        embs.append(state_emb[:, None, :])  # 添加序列维度 [B, 1, D_expert]
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks += [1]  # 新的注意力段
```

**PI0.5**：
- **不添加状态 token**
- 状态信息已经在 tokenized_prompt 中编码（作为离散 token）

#### 3.5 Suffix 组合

```python
    # 将动作 tokens 添加到嵌入列表
    embs.append(action_time_emb)
    
    # 动作 tokens 的注意力模式：第一个动作 token 开始新段，后续 tokens 在同一段内
    att_masks += [1] + ([0] * (self.config.action_horizon - 1))
    
    # 合并
    embs = torch.cat(embs, dim=1)       # [B, N_suffix, D]
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
```

**Suffix 最终维度（PI0.5）**：
- `suffix_embs`: `[B, 50, D_expert]` (只有动作 tokens，无状态 token)
- `suffix_pad_masks`: `[B, 50]`
- `suffix_att_masks`: `[B, 50]` ([1, 0, 0, ..., 0] = 因果注意力)
- `adarms_cond`: `[B, D_expert]` (时间条件)

**Suffix 最终维度（标准 PI0）**：
- `suffix_embs`: `[B, 51, D_expert]` (1 个状态 token + 50 个动作 tokens)
- `adarms_cond`: None

---

## VLM 处理阶段

### 注意力掩码构造

来源：[`src/openpi/models_pytorch/pi0_pytorch.py#L552-L620`](src/openpi/models_pytorch/pi0_pytorch.py#L552-L620)

```python
def forward(self, observation, actions, noise=None, time=None):
    """模型训练前向传播：计算流匹配损失。"""
    
    # 1-5. 前面的步骤...
    
    # 6. 组合前缀和后缀的掩码
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    # 7. 构造 2D 注意力掩码
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
```

**注意力模式**（PI0.5）：
```
                  Prefix (768+L)        Suffix (50)
                  ┌──────┬─────┐       ┌──────────┐
Prefix (768+L)    │  ✓✓  │ ✓✓  │       │    ✗✗    │
                  │  ✓✓  │ ✓✓  │       │    ✗✗    │
                  └──────┴─────┘       └──────────┘
                  
Suffix (50)       │  ✓✓  │ ✓✓  │       │ Causal   │
                  │  ✓✓  │ ✓✓  │       │   ✓      │
                  └──────┴─────┘       └─  ✓✓     ┘
```

- ✓✓：全连接注意力（Prefix 内部）
- ✓：Suffix 可以看 Prefix
- Causal：Suffix 内部因果注意力
- ✗✗：Prefix 不能看 Suffix

### Transformer 前向传播

```python
    # 10. Transformer 前向传播
    def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],  # PI0.5: adaRMS 仅用于 Expert
        )
        return suffix_out

    suffix_out = self._apply_checkpoint(forward_func, ...)
```

**维度变化**：
- 输入：
  - `prefix_embs`: `[B, N_prefix, D]`
  - `suffix_embs`: `[B, 50, D_expert]`
  - `adarms_cond`: `[B, D_expert]` (PI0.5 only)

- PaliGemma 处理：
  - Prefix → PaliGemma Transformer → `prefix_out [B, N_prefix, D]`
  
- Expert Gemma 处理（PI0.5）：
  - Suffix → **Expert Transformer with adaRMS** → `suffix_out [B, 50, D_expert]`
  - adaRMS 在每个 Transformer 层注入时间信息

**adaRMS 机制**（PI0.5 特有）：

在每个 Expert Gemma 层中：
```python
# 伪代码
hidden_states = layer_norm(hidden_states, scale=f(adarms_cond))
# f(adarms_cond) 根据时间步动态调整归一化的缩放因子
```

这使得时间信息能够直接影响 Transformer 层的归一化过程，而非简单地与输入拼接。

---

## Flow Matching 阶段

### 训练损失计算

来源：[`src/openpi/models_pytorch/pi0_pytorch.py#L552-L650`](src/openpi/models_pytorch/pi0_pytorch.py#L552-L650)

```python
def forward(self, observation, actions, noise=None, time=None):
    """模型训练前向传播：计算流匹配损失。
    
    数学原理：
    - 给定时间 t ∈ [0,1] 和噪声 ε，构造混合状态：x_t = t*ε + (1-t)*x
    - 目标速度场：u_t = ε - x
    - 训练模型预测：v_t ≈ u_t
    """
    
    # 1-3. 预处理和采样...
    
    # 4. 构造流匹配目标
    time_expanded = time[:, None, None]  # [B, 1, 1]
    x_t = time_expanded * noise + (1 - time_expanded) * actions  # 混合状态
    u_t = noise - actions  # 目标速度场
    
    # 5-10. 编码和 Transformer 前向...
    
    # 11. 提取动作输出
    suffix_out = suffix_out[:, -self.config.action_horizon :]  # [B, 50, D_expert]
    suffix_out = suffix_out.to(dtype=torch.float32)
    
    # 12. 预测速度场
    v_t = self.action_out_proj(suffix_out)  # [B, 50, 32]
    
    # 13. 计算损失
    return F.mse_loss(u_t, v_t, reduction="none")  # [B, 50, 32]
```

**Flow Matching 数学**：

$$
\begin{align}
x_t &= t \cdot \epsilon + (1-t) \cdot x_0 \quad &\text{(插值路径)} \\
u_t &= \epsilon - x_0 \quad &\text{(速度场)} \\
v_\theta(x_t, t) &\approx u_t \quad &\text{(模型预测)} \\
\mathcal{L} &= \mathbb{E}_{t, \epsilon} \|v_\theta(x_t, t) - u_t\|^2 \quad &\text{(训练损失)}
\end{align}
$$

**维度总结**：
- `actions` (目标): `[B, 50, 32]`
- `noise` (采样): `[B, 50, 32]`
- `time` (采样): `[B]`
- `x_t` (混合): `[B, 50, 32]`
- `u_t` (速度场): `[B, 50, 32]`
- `v_t` (预测): `[B, 50, 32]`
- `loss`: `[B, 50, 32]` → 通常 reduce 为标量

---

## 推理过程

### KV Cache 优化

来源：[`src/openpi/models_pytorch/pi0_pytorch.py#L652-L730`](src/openpi/models_pytorch/pi0_pytorch.py#L652-L730)

```python
@torch.no_grad()
def sample_actions(self, device, observation, noise=None, num_steps=10):
    """模型推理：从噪声生成动作序列。
    
    关键优化：
    - 使用 KV 缓存加速：前缀部分（图像+语言）只算一次
    - 每步只对后缀部分（动作）进行前向传播
    """
    
    # 1-2. 初始化和预处理...
    
    # 3. 编码前缀并构建 KV 缓存
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
    
    # 4. 生成前缀部分的 KV 缓存（仅算一次）
    _, past_key_values = self.paligemma_with_expert.forward(
        inputs_embeds=[prefix_embs, None],  # 只处理前缀
        use_cache=True,  # 开启 KV 缓存
    )
```

**KV Cache 维度**：
- `past_key_values`: List of (key, value) pairs for each layer
- 每层：
  - `key`: `[B, num_heads, N_prefix, head_dim]`
  - `value`: `[B, num_heads, N_prefix, head_dim]`

### Euler 积分求解 ODE

```python
    # 5. 设置 Euler 积分参数
    dt = -1.0 / num_steps  # 负数：从 t=1 积分到 t=0
    
    # 6. Euler 积分循环
    x_t = noise  # 从纯噪声开始
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        
        # 计算当前时刻的速度场
        v_t = self.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,  # 复用缓存
            x_t,
            expanded_time,
        )
        
        # Euler 更新
        x_t = x_t + dt * v_t
        time += dt
        
    return x_t  # 最终的动作
```

**ODE 求解**：

$$
\frac{dx}{dt} = v_\theta(x_t, t)
$$

离散化（Euler 方法）：

$$
x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)
$$

**推理路径**：
- $t=1.0$: $x_1 = \epsilon$ (纯噪声) `[B, 50, 32]`
- $t=0.9$: $x_{0.9} = x_1 + (-0.1) \cdot v_\theta(x_1, 1.0)$
- ...
- $t=0.0$: $x_0 \approx$ 目标动作 `[B, 50, 32]`

### 单步去噪

来源：[`src/openpi/models_pytorch/pi0_pytorch.py#L732-L792`](src/openpi/models_pytorch/pi0_pytorch.py#L732-L792)

```python
def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
    """执行单步去噪：预测当前时刻 x_t 的速度场 v_t。"""
    
    # 1. 编码后缀（动作+时间）
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = \
        self.embed_suffix(state, x_t, timestep)
    
    # 2. 构造组合注意力掩码
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(...)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    
    # 3. 前向传播（使用 KV 缓存）
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        past_key_values=past_key_values,  # 复用前缀的 KV 缓存
        inputs_embeds=[None, suffix_embs],    # 只处理后缀
        adarms_cond=[None, adarms_cond],      # PI0.5: adaRMS 条件
    )
    
    # 4. 提取并投影
    suffix_out = outputs_embeds[1][:, -50:]  # [B, 50, D_expert]
    return self.action_out_proj(suffix_out)  # [B, 50, 32]
```

**每步计算量**：
- **不使用 KV Cache**：需要处理全部 N_prefix + 50 个 tokens
- **使用 KV Cache**：只需处理 50 个 suffix tokens

对于 10 步推理：
- 总计算量降低约：$\frac{10 \times 50}{(768+L) + 10 \times 50} \approx$ 30-40% 的原始计算量

---

## 完整维度总结表

### 训练阶段

| 阶段 | 数据 | 维度 | 说明 |
|------|------|------|------|
| **输入** |
| | `images` | 3 × `[B, 224, 224, 3]` | 三视角 RGB 图像 |
| | `state` | `[B, 32]` | 机器人状态（PI0.5 在 token 中） |
| | `tokenized_prompt` | `[B, L]` | 指令 tokens (L≤200) |
| | `actions` | `[B, 50, 32]` | 目标动作序列 |
| **Prefix (VLM)** |
| | 图像编码 | 3 × `[B, 256, D]` → `[B, 768, D]` | SigLIP: 每图 256 patches |
| | 语言编码 | `[B, L, D]` | PaliGemma embedding |
| | `prefix_embs` | `[B, 768+L, D]` | 组合 prefix (D=2048) |
| **Flow Matching** |
| | `time` | `[B]` ∈ [0.001, 1.0] | Beta(1.5, 1.0) 采样 |
| | `noise` | `[B, 50, 32]` | N(0, I) 采样 |
| | `x_t` | `[B, 50, 32]` | 混合状态 |
| | `time_emb` | `[B, D_expert]` | 时间编码 (D_expert=1024) |
| **Suffix (Expert)** |
| | `action_emb` | `[B, 50, D_expert]` | 动作编码 |
| | `adarms_cond` (PI0.5) | `[B, D_expert]` | 时间条件 |
| | `suffix_embs` | `[B, 50, D_expert]` | PI0.5: 无状态 token |
| **Transformer** |
| | 全序列 | `[B, 768+L+50, D_expert]` | 组合 tokens |
| | 注意力掩码 | `[B, 1, 768+L+50, 768+L+50]` | 2D 掩码 |
| | `suffix_out` | `[B, 50, D_expert]` | Expert 输出 |
| **输出** |
| | `v_t` | `[B, 50, 32]` | 预测速度场 |
| | `loss` | `[B, 50, 32]` | MSE(v_t, u_t) |

### 推理阶段

| 阶段 | 数据 | 维度 | 说明 |
|------|------|------|------|
| **初始化** |
| | `x_0` | `[B, 50, 32]` | 初始噪声 t=1.0 |
| **KV Cache（一次）** |
| | `prefix_embs` | `[B, 768+L, D]` | 前缀编码 |
| | `past_key_values` | List[(K, V)] per layer | K,V: `[B, H, 768+L, D/H]` |
| **每步迭代（10次）** |
| | `x_t` | `[B, 50, 32]` | 当前状态 |
| | `time` | scalar | 当前时间步 |
| | `suffix_embs` | `[B, 50, D_expert]` | 后缀编码 |
| | `adarms_cond` | `[B, D_expert]` | PI0.5 时间条件 |
| | `v_t` | `[B, 50, 32]` | 速度场预测 |
| | 更新 | `x_t + dt * v_t` | Euler 步进 |
| **最终输出** |
| | `actions` | `[B, 50, 32]` | 去噪声的动作序列 |

---

## 关键洞察

### 1. PI0.5 的架构优势

**分离的时间注入机制**：
- 标准 PI0：时间和动作通过 MLP 融合 → 每层都看到静态的融合特征
- PI0.5：时间通过 adaRMS 动态注入 → 每层根据时间动态调整归一化

**效果**：
- 更好的时间条件控制
- 更灵活的特征调制
- 理论上更强的泛化能力

### 2. VLM 到 Flow Matching 的桥梁

**VLM 部分**（Observation → Visual-Linguistic Features）：
- 输入：多模态观测（图像、语言、状态）
- 输出：高维语义特征 `[B, N_prefix, D]`
- 作用：理解任务和环境上下文

**Flow Matching 部分**（Features + Time → Actions）：
- 输入：VLM 特征 + 噪声动作 + 时间步
- 输出：速度场预测
- 作用：将语义理解转化为具体动作

**连接点**：
- Suffix tokens 可以 attend to Prefix tokens
- Prefix 提供上下文，Suffix 生成动作
- adaRMS 将时间信息注入整个过程

### 3. 推理效率优化

**KV Cache 的价值**：
- Prefix 占总 tokens 的 ~90%（768+L vs 50）
- 缓存后每步只需计算 10% 的 tokens
- 10 步推理 → ~3-4倍加速

**代价**：
- 显存占用增加（存储 KV cache）
- 适合批次较小的推理场景

---

## 参考文献

1. **模型定义**：
   - [`src/openpi/models/pi0.py`](src/openpi/models/pi0.py)
   - [`src/openpi/models_pytorch/pi0_pytorch.py`](src/openpi/models_pytorch/pi0_pytorch.py)

2. **配置**：
   - [`src/openpi/models/pi0_config.py`](src/openpi/models/pi0_config.py)
   - [`src/openpi/models/model.py`](src/openpi/models/model.py)

3. **训练配置**：
   - [`src/openpi/training/config.py#L714-L720`](src/openpi/training/config.py#L714-L720) (PI0.5 Aloha)
   - [`src/openpi/training/config.py#L776-L785`](src/openpi/training/config.py#L776-L785) (PI0.5 DROID)

4. **预处理**：
   - [`src/openpi/models_pytorch/preprocessing_pytorch.py`](src/openpi/models_pytorch/preprocessing_pytorch.py)

---

**文档版本**：2026年1月15日  
**作者**：基于 OpenPI 代码库自动生成
