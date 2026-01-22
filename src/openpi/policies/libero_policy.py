"""
LIBERO 策略输入输出转换模块

此模块定义了 LIBERO 基准测试所需的数据转换函数，用于：
1. 将观察数据转换为模型输入格式（训练和推理）
2. 将模型输出转换回动作格式（仅推理）

主要组件：
- LiberoInputs: 输入数据转换器，处理图像、状态和语言指令
- LiberoOutputs: 输出数据转换器，提取有效动作维度
- make_libero_example: 创建随机测试样本
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """
    创建 Libero 策略的随机输入示例。
    
    用于测试和调试，生成符合 LIBERO 格式的随机数据。
    
    返回值：
        dict: 包含以下键的字典：
            - observation/state: (8,) 机器人状态（末端执行器位置、姿态、夹爪）
            - observation/image: (224, 224, 3) 第三人称视角 RGB 图像
            - observation/wrist_image: (224, 224, 3) 腕部相机 RGB 图像
            - prompt: str 任务描述文本
    """
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """
    解析和标准化图像格式。将图像转换为 (H, W, C) 格式的 uint8 数组。
    
    处理两种常见情况：
    1. float32 图像（LeRobot 格式）：归一化到 [0, 255]
    2. CHW 格式图像：转换为 HWC 格式
    
    参数：
        image: 输入图像，可以是 float32 或 uint8，CHW 或 HWC 格式
    
    返回值：
        np.ndarray: (H, W, C) 格式的 uint8 图像数组
    """
    image = np.asarray(image)
    # 如果是浮点数图像，转换为 uint8
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # 如果是 CHW 格式，转换为 HWC
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    LIBERO 输入数据转换器
    
    将原始观察数据转换为模型期望的输入格式，用于训练和推理阶段。
    
    主要功能：
    1. 解析和标准化图像格式（uint8, HWC）
    2. 组织多视角图像输入（第三人称 + 左腕 + 右腕）
    3. 处理机器人状态向量
    4. 传递语言指令（prompt）
    5. 添加图像掩码以标识有效/填充图像
    
    对于自定义数据集：
    - 可以复制此类并根据注释修改键名
    - 确保输出字典的键名保持不变（模型要求）
    - 调整图像视角配置以匹配数据集
    
    属性：
        model_type: 使用的模型类型（PI0 或 PI0_FAST），影响掩码策略
    """

    # 决定使用哪个模型。对于自定义数据集，请勿更改此项。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """
        执行数据转换
        
        将数据集格式的观察数据转换为模型输入格式。
        
        参数：
            data: 包含观察数据的字典，键包括：
                - observation/image: 第三人称视角图像
                - observation/wrist_image: 腕部相机图像
                - observation/state: 机器人状态向量（8维）
                - prompt: 任务描述（可选）
                - actions: 动作序列（仅训练时）
        
        返回值：
            dict: 模型输入格式的字典，包含：
                - state: 机器人状态
                - image: 多视角图像字典
                - image_mask: 图像有效性掩码
                - prompt: 任务指令（如果有）
                - actions: 动作标签（仅训练时）
        """
        # 可能需要将图像解析为 uint8 (H,W,C) 格式，因为 LeRobot 自动存储为 float32 (C,H,W)
        # 在策略推理时会跳过此步骤。
        # 对于自定义数据集保留此部分，但如果数据集将图像存储在不同的键中
        # （而不是 "observation/image" 或 "observation/wrist_image"），应在下面更改键名。
        #
        # Pi0 模型目前支持三个图像输入：一个第三人称视角，两个腕部视角（左和右）。
        # 如果数据集没有特定类型的图像（例如腕部图像），可以在此注释掉并用零数组替换，
        # 就像我们在下面对右腕图像所做的那样。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 创建输入字典。请勿更改下面字典中的键名（模型要求）。
        inputs = {
            # 机器人状态：末端执行器位置(3) + 四元数(4) + 夹爪(1) = 8维
            "state": data["observation/state"],
            
            # 多视角图像输入
            "image": {
                "base_0_rgb": base_image,          # 第三人称固定相机
                "left_wrist_0_rgb": wrist_image,   # 左腕部相机
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 用零数组填充不存在的右腕相机
            },
            
            # 图像掩码：标识哪些图像是有效的（True）或填充的（False）
            "image_mask": {
                "base_0_rgb": np.True_,           # 第三人称图像始终有效，np.True_表示布尔值 True，和True的区别是类型为np.bool_，True是内置类型bool
                "left_wrist_0_rgb": np.True_,     # 左腕图像始终有效
                # 我们只为 pi0 模型掩码填充图像，pi0-FAST 不需要。对于自定义数据集，请勿更改此项。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 将动作填充到模型动作维度。对于自定义数据集保留此部分。
        # 动作仅在训练期间可用。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 将提示（即语言指令）传递给模型。
        # 对于自定义数据集保留此部分（但如果指令未存储在 "prompt" 中，请修改键名；
        # 输出字典始终需要有 "prompt" 键）。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    LIBERO 输出数据转换器
    
    将模型输出转换回数据集特定格式。仅用于推理阶段。
    
    主要功能：
    1. 从模型输出中提取有效动作维度
    2. 移除填充动作（因为模型可能输出更高维度）
    
    对于自定义数据集：
    - 可以复制此类并根据注释修改动作维度
    - 确保返回的动作维度与环境期望的动作空间匹配
    
    LIBERO 动作空间（7 维）：
    - 末端执行器位置增量 (3): delta_x, delta_y, delta_z
    - 末端执行器旋转增量 (3): delta_roll, delta_pitch, delta_yaw
    - 夹爪动作 (1): open/close [-1, 1]
    """

    def __call__(self, data: dict) -> dict:
        """
        执行输出转换
        
        从模型输出中提取 LIBERO 所需的 7 维动作。
        
        参数：
            data: 包含模型输出的字典，键包括：
                - actions: 形状为 (batch, action_dim) 的动作数组
        
        返回值：
            dict: 包含提取后的动作：
                - actions: 形状为 (batch, 7) 的动作数组
        """
        # 仅返回前 N 维度的动作 -- 由于我们在上面填充了动作以适应模型动作维度，
        # 现在需要从返回字典中解析出正确数量的动作。
        # 对于 Libero，我们仅返回前 7 维度的动作（因为其余部分是填充）。
        # 对于自定义数据集，将 `7` 替换为数据集的动作维度。
        # shape data["actions"] = (batch, action_dim)
        return {"actions": np.asarray(data["actions"][:, :7])}
