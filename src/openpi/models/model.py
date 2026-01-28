"""OpenPI模型核心模块。

本模块定义了OpenPI框架的核心模型抽象和数据结构，包括：

1. 模型基础类和配置
   - BaseModel: 所有PI模型的抽象基类
   - BaseModelConfig: 模型配置的基础类
   - ModelType: 支持的模型类型枚举

2. 数据结构定义
   - Observation: 机器人观测数据结构（图像、状态、提示等）
   - Actions: 动作序列数据结构
   - 标准化的输入/输出格式

3. 模型操作工具
   - 模型加载和保存
   - 权重参数的恢复和管理
   - JAX/PyTorch模型间的转换

4. 支持的模型类型
   - PI0: 标准流匹配扩散模型
   - PI0_FAST: 自回归版本的快速推理模型
   - PI05: 使用adaRMS归一化的改进版本

关键特性：
- 多模态输入支持（图像、状态、文本）
- 灵活的动作序列预测
- 跨框架模型兼容性（JAX/PyTorch）
- 标准化的数据接口
- 分布式训练支持

数据格式约定：
- 图像分辨率：224x224像素
- 标准图像键：base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
- 动作序列：时间序列形式的控制指令
- 状态表示：关节角度、夹爪位置等机器人状态信息
"""

import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# 数组类型的类型变量（JAX数组、PyTorch张量或numpy数组）
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """支持的模型类型。
    
    OpenPI框架支持多种模型架构的变体：
    - PI0: 标准的流匹配扩散模型，用于连续动作预测
    - PI0_FAST: 自回归版本，支持更快的推理速度
    - PI05: 改进版本，使用adaRMS归一化技术
    """

    PI0 = "pi0"          # 标准流匹配模型
    PI0_FAST = "pi0_fast"  # 快速自回归模型
    PI05 = "pi05"        # 改进版本（adaRMS）


# 模型始终期望的图像键名
# 这些对应不同摄像头的标准命名约定：
# - base_0_rgb: 主摄像头（通常是固定的外部视角）
# - left_wrist_0_rgb: 左臂腕部摄像头
# - right_wrist_0_rgb: 右臂腕部摄像头
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb", 
    "right_wrist_0_rgb",
)


# 标准图像分辨率
# 如果发布小型模型，此参数可能需要调整
IMAGE_RESOLUTION = (224, 224)


# 数据格式说明
#
# 数据变换器产生模型输入，格式为嵌套字典，稍后转换为 `Observation` 和 `Actions` 对象。
# 详见下方定义。
#
# 在字典形式中，数据应该如下所示：
# {
#     # 观测数据
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB图像，范围[-1, 1]或[0, 255]
#         ...  # 其他摄像头视角
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True表示图像有效
#         ...  # 其他视角的掩码
#     },
#     "state": float32[*b, s],  # 低维机器人状态
#     "tokenized_prompt": int32[*b, l],  # 可选，分词后的语言提示
#     "tokenized_prompt_mask": bool[*b, l],  # 可选，分词提示的掩码
#     "token_ar_mask": int32[*b, l],  # 可选，FAST模型的自回归掩码
#     "token_loss_mask": bool[*b, l],  # 可选，FAST模型的损失掩码
#
#      # 动作数据
#      "actions": float32[*b ah ad]
# }
# 其中：
#   *b = 批次维度
#   h,w = 图像高度/宽度
#   s = 状态维度
#   l = 序列长度
#   ah = 动作序列长度
#   ad = 动作维度
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """保存观测数据，即模型的输入。
    
    观测数据包含机器人感知到的所有信息：
    - 多摄像头图像：提供视觉信息，用于理解环境和物体
    - 机器人状态：关节角度、夹爪位置等本体感觉信息
    - 语言提示：任务描述，指导机器人执行特定行为
    
    数据格式要求：
    - 图像：标准化到[-1,1]范围的float32格式
    - 状态：归一化的低维向量
    - 提示：经过分词器处理的token序列
    
    参见 `Observation.from_dict` 了解期望的字典格式。
    这是数据变换器应该产生的格式。
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

    # pi0-fast 模型特定字段（用于自回归模型）

    # Token 自回归掩码（用于 FAST 自回归模型）
    # 控制每个 token 可以关注哪些之前的 token
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token 损失掩码（用于 FAST 自回归模型）
    # 标记哪些 token 需要计算损失
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """从嵌套字典创建 Observation 对象。
        
        该方法定义了非结构化数据（嵌套字典）到结构化 Observation 格式的映射。
        这是数据变换器输出和模型输入之间的标准接口。
        
        数据预处理：
        - 确保 tokenized_prompt 和 tokenized_prompt_mask 成对出现
        - 自动转换 uint8 图像到 float32 范围 [-1, 1]
        - 处理不同框架的图像格式（JAX/PyTorch）
        
        Args:
            data: 包含观测数据的嵌套字典，应包含以下键：
                - "image": 图像字典
                - "image_mask": 图像掩码字典
                - "state": 机器人状态
                - "tokenized_prompt" (可选): 分词后的提示
                - "tokenized_prompt_mask" (可选): 提示掩码
                
        Returns:
            结构化的 Observation 对象
            
        Raises:
            ValueError: 如果 tokenized_prompt 和 tokenized_prompt_mask 只提供了其中一个
        """
        # 确保 tokenized_prompt 和 tokenized_prompt_mask 成对提供
        # 语言提示的 token 序列和掩码必须同时存在或同时不存在
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        
        # 如果图像是 uint8 格式（[0, 255]），转换为 float32 范围 [-1, 1]
        # 这是模型期望的标准输入格式
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                # NumPy/JAX 格式：[B, H, W, C] uint8 -> [B, H, W, C] float32
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                # PyTorch 格式：[B, C, H, W] uint8 -> [B, H, W, C] float32
                # 注意：PyTorch 通常使用 NCHW 格式，需要转置为 NHWC
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """将 Observation 对象转换为嵌套字典。
        
        这是 from_dict 的逆操作，用于：
        - 保存数据到磁盘
        - 与其他系统交互
        - 调试和可视化
        
        键名映射：
        - 'images' -> 'image'
        - 'image_masks' -> 'image_mask'
        
        Returns:
            包含所有观测数据的嵌套字典
        """
        result = dataclasses.asdict(self)
        # 将内部键名转换为外部标准键名
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# 定义动作序列的格式
# 该字段在数据变换器产生的字典中以 "actions" 键包含
# 维度说明：
#   *b: 批次维度（可以是多维的，如 [batch] 或 [batch, num_envs]）
#   ah: action_horizon（动作序列长度，通常为 50）
#   ad: action_dim（动作维度，通常为 32，对应机器人的关节数+夹爪）
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """预处理观测数据，包括图像增强、调整大小和掩码处理。
    
    该函数执行以下预处理步骤：
    1. 验证所需的图像键是否存在
    2. 调整图像分辨率（如果需要）
    3. 应用数据增强（仅训练模式）
    4. 填充默认的图像掩码（如果缺失）
    
    训练时的数据增强包括：
    - 随机裁剪（95% 大小）：轻微的空间扰动
    - 随机旋转（±5°）：仅应用于非腕部摄像头
    - 颜色抖动：调整亮度、对比度和饱和度
    
    Args:
        rng: JAX 随机数生成器密钥，用于数据增强（训练模式需要）
        observation: 原始观测数据
        train: 是否为训练模式，True 时应用数据增强
        image_keys: 需要处理的图像键列表
        image_resolution: 目标图像分辨率 (height, width)
        
    Returns:
        预处理后的 Observation 对象，所有图像都符合标准格式
        
    Raises:
        ValueError: 如果 observation.images 缺少必需的图像键
    """

    # 验证所有必需的图像键都存在
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # 从状态张量推断批次形状（去除最后一维，即状态维度）
    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key] # shape: [*b, h, w, c]
        # 检查图像分辨率是否匹配，不匹配则调整
        # image.shape[1:3] 对应 (height, width)，跳过批次维度
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            # 使用带填充的调整大小，保持纵横比
            image = image_tools.resize_with_pad(image, *image_resolution) # shape: [*b, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], c]

        if train:
            # 将图像从 [-1, 1] 转换到 [0, 1] 范围，因为 augmax 期望此格式
            image = image / 2.0 + 0.5

            transforms = []
            # 对于非腕部摄像头（base_0_rgb 等），应用空间变换
            # 腕部摄像头通常更近距离，不适合旋转和裁剪
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    # 随机裁剪到 95% 大小，引入轻微的空间扰动
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    # 调整回原始大小
                    augmax.Resize(width, height),
                    # 随机旋转 ±5 度，增加方向多样性
                    augmax.Rotate((-5, 5)),
                ]
            # 对所有摄像头应用颜色抖动，增强对光照变化的鲁棒性
            transforms += [
                augmax.ColorJitter(
                    brightness=0.3,  # 亮度抖动范围
                    contrast=0.4,    # 对比度抖动范围
                    saturation=0.5,  # 饱和度抖动范围
                ),
            ]
            # 为批次中的每个样本分配独立的随机数生成器
            sub_rngs = jax.random.split(rng, image.shape[0])
            # 使用 vmap 并行应用变换到批次中的所有图像
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 转换回 [-1, 1] 范围，这是模型期望的输入格式
            image = image * 2.0 - 1.0

        out_images[key] = image

    # 获取或创建图像掩码
    # 掩码用于标识哪些图像是有效的（例如，某些摄像头可能暂时不可用）
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 如果没有提供掩码，默认所有图像都有效（全为 True）
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool) 
            # out_masks shape: [key_dim, *b]
        else:
            # 使用提供的掩码，确保是 JAX 数组格式
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """所有模型共享的基础配置类。
    
    特定模型应继承此类，并实现 `create` 方法来创建相应的模型实例。
    这个抽象基类定义了所有 OpenPI 模型的通用接口和配置参数。
    
    设计模式：
    - 使用 dataclass 提供便捷的配置管理
    - frozen=True 确保配置不可变
    - 抽象方法强制子类实现必要的功能
    
    子类示例：Pi0Config, Pi0FastConfig
    """

    # 动作空间维度（通常为 32，对应机器人的关节数+夹爪）
    action_dim: int
    # 动作序列长度（通常为 50，表示预测未来 50 步的动作）
    action_horizon: int
    # 分词后提示的最大长度（PI0: 48, PI0.5: 200）
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """返回模型类型（PI0、PI0_FAST 或 PI05）。
        
        子类必须实现此属性来标识其模型类型。
        """

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """创建新模型并初始化参数。
        
        使用提供的随机数生成器初始化模型的所有可学习参数。
        这是训练新模型时的入口点。
        
        Args:
            rng: JAX 随机数生成器密钥，用于参数初始化
            
        Returns:
            完全初始化的模型实例
        """

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """使用给定的参数创建模型（加载预训练模型）。
        
        从检查点或预训练权重加载模型参数。这个方法：
        1. 创建模型结构（不初始化参数）
        2. 验证参数形状匹配
        3. 加载提供的参数
        
        Args:
            params: 参数字典，通常从检查点加载
            remove_extra_params: 是否移除检查点中多余的参数
                True: 只加载与模型结构匹配的参数（推荐）
                False: 要求检查点参数完全匹配模型结构
                
        Returns:
            加载了预训练参数的模型实例
            
        Raises:
            AssertionError: 如果参数形状不匹配
        """
        # 创建模型结构但不实际初始化参数（eval_shape 只计算形状）
        model = nnx.eval_shape(self.create, jax.random.key(0))
        # 将模型分离为图定义（结构）和状态（参数）
        graphdef, state = nnx.split(model)
        
        if remove_extra_params:
            # 只保留与模型状态匹配的参数，忽略检查点中的额外参数
            # 这在加载部分预训练模型或跨版本兼容时很有用
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        
        # 验证参数树的结构和形状是否匹配（但允许数据类型不同）
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        # 用加载的参数替换模型状态
        state.replace_by_pure_dict(params)
        # 重新组合图定义和状态，得到完整的模型
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        """加载 PyTorch 版本的模型。
        
        从 safetensors 格式的权重文件加载 PyTorch 实现的模型。
        用于推理部署或跨框架模型转换。
        
        Args:
            train_config: 训练配置对象，包含模型配置
            weight_path: safetensors 权重文件的路径
            
        Returns:
            加载了预训练权重的 PyTorch 模型
        """
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """返回模型的输入规格。
        
        提供模型期望的输入数据的形状和数据类型信息，
        使用 jax.ShapeDtypeStruct 表示（不包含实际数据）。
        
        这对于以下场景很有用：
        - JIT 编译前确定输入形状
        - 创建测试数据
        - 验证数据管道输出
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (observation_spec, action_spec) 元组，包含输入规格
        """

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """创建虚拟观测数据用于测试。
        
        生成符合模型输入规格的全 1 张量，用于：
        - 模型结构验证
        - JIT 编译预热
        - 单元测试
        
        Args:
            batch_size: 批次大小
            
        Returns:
            填充了全 1 的虚拟 Observation 对象
        """
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        """创建虚拟动作数据用于测试。
        
        生成符合模型输出规格的全 1 张量，用途同 fake_obs。
        
        Args:
            batch_size: 批次大小
            
        Returns:
            填充了全 1 的虚拟 Actions 张量
        """
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """所有模型实现的抽象基类。
    
    特定模型应继承此类，并实现抽象方法 compute_loss 和 sample_actions。
    子类在初始化时应调用 super().__init__() 来初始化共享属性。
    
    职责：
    - 定义模型的标准接口（损失计算和动作采样）
    - 存储模型的基本配置（动作维度、序列长度等）
    - 提供统一的模型交互方式
    
    继承自：
    - nnx.Module: Flax NNX 模块，提供参数管理和 JIT 支持
    - abc.ABC: 抽象基类，强制子类实现必要方法
    
    子类示例：Pi0, Pi0Fast
    """

    action_dim: int          # 动作空间维度
    action_horizon: int      # 动作序列长度
    max_token_len: int       # 最大 token 长度

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,pi
    ) -> at.Float[at.Array, "*b ah"]:
        """计算模型的训练损失。
        
        给定观测和目标动作，计算预测的损失值。
        不同模型可能使用不同的损失函数：
        - PI0/PI0.5: Flow Matching MSE 损失
        - PI0_FAST: 自回归交叉熵损失
        
        Args:
            rng: 随机数生成器，用于噪声采样等随机操作
            observation: 输入观测数据
            actions: 目标动作序列
            train: 是否为训练模式（影响数据增强、dropout 等）
            
        Returns:
            损失张量，形状为 [*batch, action_horizon]
            通常在外部进一步 reduce（求和或平均）
        """

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions:
        """从观测中采样生成动作序列。
        
        给定当前观测，生成要执行的动作序列。
        这是推理时的主要接口。
        
        不同模型的采样策略：
        - PI0/PI0.5: 通过 ODE 求解器从噪声去噪到动作
        - PI0_FAST: 自回归解码
        
        Args:
            rng: 随机数生成器
            observation: 输入观测数据
            **kwargs: 模型特定的参数（如 num_steps, temperature 等）
            
        Returns:
            生成的动作序列，形状为 [batch, action_horizon, action_dim]
        """
        ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """从检查点恢复非结构化的参数 PyTree。

    这个函数可以加载两种类型的检查点：
    1. OpenPI 训练过程中保存的检查点（通过 save_state，见 training/checkpoints.py）
    2. 官方发布的预训练检查点
    
    功能特性：
    - 支持本地路径和 Google Cloud Storage (gs://) 路径
    - 灵活的数据类型和分片控制
    - 自动处理 NNX State 的 "value" 后缀
    - 支持分布式训练的参数分片
    
    使用场景：
    - 加载预训练模型进行微调
    - 从检查点恢复训练
    - 模型评估和推理

    Args:
        params_path: 检查点目录的路径（本地或 gs:// 路径）
        restore_type: 恢复参数的数据类型
            - jax.Array (默认): 在 GPU/TPU 上加载为 JAX 数组
            - np.ndarray: 加载为 NumPy 数组（用于 CPU 或序列化）
        dtype: 恢复所有参数的数据类型
            - None (默认): 使用检查点中的原始数据类型
            - 指定 dtype: 将所有参数转换为该类型（如 jnp.float32）
        sharding: 参数的分片策略（用于分布式训练）
            - None (默认): 参数在所有设备上复制
            - 指定 Sharding: 按指定策略分片参数

    Returns:
        恢复的参数字典，格式为 "pure dict"（NNX 术语，不含 "value" 后缀）
        
    示例：
        >>> # 加载预训练模型
        >>> params = restore_params("gs://openpi-assets/checkpoints/pi05_base/params")
        >>> # 加载并转换数据类型
        >>> params = restore_params("./checkpoint", dtype=jnp.float16)
        >>> # 加载为 NumPy 数组
        >>> params = restore_params("./checkpoint", restore_type=np.ndarray)
    """
    # 处理路径：本地路径解析为绝对路径，GCS 路径保持原样
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    # 如果恢复为 JAX 数组且未指定分片，创建默认的复制分片
    # 这确保参数在所有设备上都可用
    if restore_type is jax.Array and sharding is None:
        # 创建一个简单的 mesh，将所有设备放在一个维度上
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        # 创建复制分片（所有设备都有完整副本）
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 使用 Orbax 检查点工具恢复参数
    with ocp.PyTreeCheckpointer() as ckptr:
        # 读取检查点元数据以获取参数结构
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        # 恢复参数，应用指定的数据类型和分片策略
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                # 为参数树的每个叶子节点设置恢复参数
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding,      # 分片策略
                        restore_type=restore_type,  # 数组类型
                        dtype=dtype             # 数据类型
                    ), 
                    item
                ),
            ),
        )["params"]

    # 如果参数是通过 OpenPI 训练中的 save_state 保存的，
    # 每个键路径都会以 "value" 结尾（由 nnx.State 添加）
    # 我们在这里移除 "value" 后缀，始终返回 NNX 所称的 "pure dict"
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        # 移除所有键路径的最后一个元素（"value"）
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    # 将扁平化的字典重新构造为嵌套结构
    return traverse_util.unflatten_dict(flat_params)
