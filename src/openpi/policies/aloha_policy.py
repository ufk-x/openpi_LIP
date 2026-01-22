"""Aloha机器人策略模块。

Aloha是一个双臂操作机器人系统，具有14个自由度：
- 每个手臂7个自由度（6个关节 + 1个夹爪）
- 支持多摄像头视觉输入（高位、低位、左腕、右腕）
- 用于桌面操作任务，如捡拾、放置、组装等

本模块提供：
1. Aloha机器人的输入/输出数据变换
2. 关节角度和夹爪控制的标准化
3. 多摄像头图像的统一处理
4. 与OpenPI训练框架的接口

关键组件：
- AlohaInputs: 处理机器人状态、图像和提示的输入变换
- AlohaOutputs: 处理预测动作序列的输出变换
- 数据适配功能：在不同的动作空间之间转换

Aloha机器人配置：
- 状态维度：14（每个手臂7个自由度）
- 动作维度：14（对应状态维度）
- 摄像头：4个（高位、低位、左腕、右腕）
- 图像分辨率：224x224像素
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """创建Aloha策略的随机输入示例。
    
    生成符合Aloha机器人预期格式的测试数据，包括：
    - 机器人状态：14维向量（双臂关节角度和夹爪位置）
    - 多摄像头图像：4个摄像头的RGB图像数据
    - 任务提示：自然语言指令
    
    这个函数主要用于：
    - 单元测试和集成测试
    - 模型推理的数据格式验证
    - 演示和调试
    
    Returns:
        包含机器人状态、图像和提示的字典，格式与真实数据一致
    """
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Aloha机器人策略的输入数据变换器。
    
    将原始的机器人数据转换为模型可以处理的标准格式。
    主要功能包括：
    - 多摄像头图像的标准化处理
    - 机器人状态（关节角度、夹爪位置）的格式化
    - 缺失摄像头数据的处理（用黑图像填充）
    - 可选的动作空间适配
    
    预期输入格式：
    - images: dict[摄像头名称, 图像数据]，图像格式为[通道, 高度, 宽度]
    - state: [14] 维状态向量（双臂关节角度和夹爪位置）
    - actions: [动作序列长度, 14] 动作序列
    
    输出格式：
    - 标准化的图像序列
    - 处理后的机器人状态
    - 图像可用性掩码
    
    注意：
    - 所有输入摄像头名称必须在EXPECTED_CAMERAS中
    - 缺失的摄像头将用黑图像替代，对应的掩码设为False
    """

    # 如果为真，将关节和夹爪值从标准Aloha空间转换为
    # 用于训练基础模型的pi内部运行时使用的空间
    adapt_to_pi: bool = True

    # 预期的摄像头名称。所有输入摄像头必须在此集合中。
    # 缺失的摄像头将被黑图像替换，对应的 `image_mask` 将设置为 False
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        """将原始 Aloha 数据转换为模型输入格式。
        
        处理流程：
        1. 解码 Aloha 原始数据（图像格式转换、状态空间适配）
        2. 验证所有摄像头名称的合法性
        3. 组织图像数据（基础摄像头 + 手腕摄像头）
        4. 为缺失的摄像头创建占位图像和掩码
        5. 处理动作序列（仅在训练时）
        6. 添加任务提示（如果有）
        
        Args:
            data: 包含 images, state, actions（可选）, prompt（可选）的字典
            
        Returns:
            标准化的模型输入字典
        """
        # 1. 解码 Aloha 数据：图像格式转换（CHW -> HWC）、状态空间适配
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        # 2. 验证输入图像的摄像头名称
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # 3. 假设基础摄像头（高位摄像头）总是存在
        base_image = in_images["cam_high"]

        # 4. 初始化图像字典和掩码字典
        # base_0_rgb: 基础视角（高位摄像头），用于全局场景观察
        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,  # 基础摄像头总是可用
        }

        # 5. 添加额外的手腕摄像头图像
        # 手腕摄像头提供近距离的末端执行器视角，对精细操作很重要
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",    # 左臂手腕摄像头
            "right_wrist_0_rgb": "cam_right_wrist",  # 右臂手腕摄像头
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                # 摄像头数据存在，直接使用
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                # 摄像头数据缺失，用黑图像填充并标记为不可用
                # 这样可以保持输入维度一致，模型可以通过掩码忽略这些图像
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # 6. 组装最终的输入字典
        inputs = {
            "image": images,           # 所有摄像头的图像数据
            "image_mask": image_masks, # 标记哪些摄像头可用
            "state": data["state"],    # 机器人当前状态（关节角度和夹爪位置）
        }

        # 7. 添加动作序列（仅在训练时可用）
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # 将动作从训练数据空间转换回 pi 内部空间
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        # 8. 添加任务提示（自然语言指令）
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Aloha 机器人策略的输出数据变换器。
    
    将模型预测的动作转换为 Aloha 机器人可以执行的格式。
    主要功能包括：
    - 提取前 14 维动作（双臂的关节角度和夹爪控制）
    - 可选的动作空间转换（从 pi 内部空间到 Aloha 标准空间）
    - 关节角度的符号翻转
    - 夹爪位置的角度到线性空间转换
    
    输入格式：
    - actions: [动作序列长度, >=14] 模型预测的动作序列
    
    输出格式：
    - actions: [动作序列长度, 14] Aloha 可执行的动作序列
    
    注意：
    - 只使用前 14 维，忽略其他维度（如果有）
    - adapt_to_pi=True 时会进行空间转换
    """

    # 如果为真，将关节和夹爪值从 pi 内部空间转换为 Aloha 标准空间
    # pi 内部空间是训练基础模型时使用的空间表示
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        """转换模型输出为 Aloha 可执行的动作。
        
        Args:
            data: 包含 'actions' 键的字典，值为预测的动作序列
            
        Returns:
            包含转换后动作的字典，可直接用于 Aloha 机器人执行
        """
        # 只返回前 14 维（双臂 7 自由度 × 2 = 14）
        # 格式：[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]
        actions = np.asarray(data["actions"][:, :14])
        
        # 如果需要，转换动作空间（关节翻转 + 夹爪角度转线性）
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """用于在 Aloha 和 pi 关节角度之间转换的符号掩码。
    
    Aloha 和 pi 内部运行时使用不同的关节角度约定：
    - 某些关节的正方向相反
    - 需要通过符号翻转来对齐两个空间
    
    掩码格式：[左臂7维, 右臂7维]
    - 1: 保持原符号
    - -1: 翻转符号
    
    具体翻转的关节：
    - 左臂：关节 1, 2（索引 1, 2）
    - 右臂：关节 1, 2（索引 8, 9）
    
    Returns:
        14维符号掩码数组，用于元素级乘法进行空间转换
    """
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    """将值从原始范围 [min_val, max_val] 归一化到 [0, 1]。
    
    归一化公式：normalized = (x - min) / (max - min)
    
    Args:
        x: 要归一化的值
        min_val: 原始范围的最小值
        max_val: 原始范围的最大值
        
    Returns:
        归一化后的值，范围在 [0, 1]
    """
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    """将值从 [0, 1] 反归一化到原始范围 [min_val, max_val]。
    
    反归一化公式：unnormalized = x * (max - min) + min
    
    Args:
        x: 归一化的值，范围在 [0, 1]
        min_val: 目标范围的最小值
        max_val: 目标范围的最大值
        
    Returns:
        反归一化后的值，范围在 [min_val, max_val]
    """
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    """将夹爪位置从 Aloha 线性空间转换为 pi0 角度空间。
    
    转换流程：
    1. 反归一化：[0, 1] -> [0.01844, 0.05800] 米（线性位置）
    2. 线性位置 -> 角度：使用逆运动学公式
    3. 归一化角度：映射到 pi0 的归一化范围
    
    背景说明：
    - Aloha 使用线性空间表示夹爪位置（基于机械臂长度）
    - pi0 使用角度空间（基于编码器读数）
    - 需要在两个空间之间转换以保持数据一致性
    
    物理参数（来自 Interbotix 代码）：
    - arm_length: 0.036 米（机械臂长度）
    - horn_radius: 0.022 米（舵机臂半径）
    - 编码器范围：2405-3110（总共 4096 个计数）
    - 零点：2048（编码器中心）
    
    Args:
        value: Aloha 归一化的夹爪位置 [0, 1]
        
    Returns:
        pi0 归一化的夹爪角度 [0, 1]，对应角度范围 [0.5476, 1.6296] 弧度
    """
    # 步骤 1：将 Aloha 归一化值转换为实际线性位置（米）
    # 这些常量来自 Aloha 代码：
    # PUPPET_GRIPPER_POSITION_OPEN = 0.01844 米（夹爪打开）
    # PUPPET_GRIPPER_POSITION_CLOSED = 0.05800 米（夹爪闭合）
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 步骤 2：线性位置转换为角度（弧度）
    # 这是 Interbotix 代码中角度到线性转换的逆过程
    def linear_to_radian(linear_position, arm_length, horn_radius):
        """使用余弦定理的逆运算将线性位置转换为角度。
        
        机械原理：
        - 夹爪通过连杆机构将舵机旋转转换为线性运动
        - 使用三角形几何关系推导角度
        
        Args:
            linear_position: 夹爪的线性位置（米）
            arm_length: 机械臂长度（米）
            horn_radius: 舵机臂半径（米）
            
        Returns:
            舵机角度（弧度）
        """
        # 余弦定理的逆运算：arcsin((r^2 + d^2 - l^2) / (2*r*d))
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        # 限制在 [-1, 1] 以确保 arcsin 有效
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # 使用 Interbotix 机械参数
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # 步骤 3：归一化到 pi0 的范围
    # pi0 夹爪数据在编码器计数 (2405, 3110) 之间归一化到 (0, 1)
    # 编码器总计数：4096，Aloha 零点：2048
    # 转换为弧度后的归一化范围：(0.5476, 1.6296) 弧度
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    """将夹爪角度从 pi0 空间转换为 Aloha 空间。
    
    这是 _gripper_to_angular 的逆过程，用于将模型预测的
    夹爪角度转换为 Aloha 机器人可以执行的格式。
    
    转换流程：
    1. 偏移调整：加上 pi0 的零点偏移 (0.5476)
    2. 归一化：映射到 Aloha 的角度范围
    
    注意：
    - 单位仍然是角度（弧度），但范围不同
    - Trossen 模型的预测已经是弧度，所以不需要额外缩放
    
    Args:
        value: pi0 的夹爪角度（已经过归一化的弧度值）
        
    Returns:
        Aloha 归一化的夹爪角度 [0, 1]
    """
    # 步骤 1：添加 pi0 的零点偏移
    # 见 _gripper_to_angular 中对该常量的推导
    value = value + 0.5476

    # 步骤 2：归一化到 Aloha 的角度范围
    # 这些常量来自 Aloha 代码：
    # PUPPET_GRIPPER_JOINT_OPEN = -0.6213 弧度（夹爪打开）
    # PUPPET_GRIPPER_JOINT_CLOSE = 1.4910 弧度（夹爪闭合）
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    """_gripper_from_angular 的精确逆函数。
    
    用于将 Aloha 归一化的夹爪角度转换回 pi0 空间。
    这在处理训练数据时需要，以便将 Aloha 动作转换为
    pi0 内部表示用于训练。
    
    Args:
        value: Aloha 归一化的夹爪角度 [0, 1]
        
    Returns:
        pi0 的夹爪角度（归一化的弧度值）
    """
    # 步骤 1：反归一化到 Aloha 的实际角度范围
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    # 步骤 2：减去 pi0 的零点偏移
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    """解码 Aloha 原始数据，转换为标准格式。
    
    主要处理：
    1. 状态向量的空间转换（如果需要）
    2. 图像格式转换（CHW -> HWC，浮点 -> uint8）
    
    状态向量结构：
    - 左臂关节角度：[0:6]（6个关节）
    - 左臂夹爪：[6]（1个值）
    - 右臂关节角度：[7:13]（6个关节）
    - 右臂夹爪：[13]（1个值）
    - 总维度：14
    
    Args:
        data: 包含 'state' 和 'images' 的原始数据字典
        adapt_to_pi: 是否将状态空间转换为 pi 内部表示
        
    Returns:
        转换后的数据字典，图像格式为 HWC，状态空间已适配
    """
    # 1. 处理状态向量
    # state 结构: [左臂6关节, 左夹爪, 右臂6关节, 右夹爪]
    # 维度大小: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    # 2. 处理图像数据
    def convert_image(img):
        """转换图像格式为标准的 HWC uint8 格式。
        
        处理：
        - 浮点图像 [0, 1] -> uint8 [0, 255]
        - CHW 格式 -> HWC 格式（适配 OpenCV/PIL）
        
        Args:
            img: 输入图像，可能是 CHW 格式的浮点或 uint8
            
        Returns:
            HWC 格式的 uint8 图像
        """
        img = np.asarray(img)
        # 如果是浮点图像，转换为 uint8（假设范围 [0, 1]）
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # 从 [通道, 高度, 宽度] 转换为 [高度, 宽度, 通道]
        # 这是大多数图像处理库（如 OpenCV）期望的格式
        return einops.rearrange(img, "c h w -> h w c")

    # 转换所有摄像头的图像
    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    # 更新数据字典
    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """解码机器人状态，从 Aloha 空间转换为 pi 内部空间。
    
    处理步骤（如果 adapt_to_pi=True）：
    1. 翻转某些关节的符号（对齐坐标系约定）
    2. 转换夹爪位置（线性空间 -> 角度空间）
    
    Args:
        state: [14] 维状态向量，Aloha 格式
        adapt_to_pi: 是否执行空间转换
        
    Returns:
        转换后的状态向量，pi 内部格式
    """
    if adapt_to_pi:
        # 1. 翻转关节符号
        # 某些关节在 Aloha 和 pi 中有相反的正方向约定
        state = _joint_flip_mask() * state
        
        # 2. 转换夹爪位置
        # 逆转 Aloha 运行时应用的夹爪变换
        # 索引 6: 左臂夹爪，索引 13: 右臂夹爪
        # 从线性空间转换为角度空间
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """编码动作，从 pi 内部空间转换为 Aloha 执行空间。
    
    这是 _decode_state 的逆过程，用于将模型预测的动作
    转换为 Aloha 机器人可以直接执行的格式。
    
    处理步骤（如果 adapt_to_pi=True）：
    1. 翻转某些关节的符号
    2. 转换夹爪角度（角度空间 -> Aloha 角度范围）
    
    Args:
        actions: [序列长度, 14] 动作序列，pi 内部格式
        adapt_to_pi: 是否执行空间转换
        
    Returns:
        转换后的动作序列，Aloha 执行格式
    """
    if adapt_to_pi:
        # 1. 翻转关节符号
        actions = _joint_flip_mask() * actions
        
        # 2. 转换夹爪角度
        # 将 pi 的夹爪角度转换为 Aloha 的归一化角度
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """编码动作的逆函数，用于处理训练数据。
    
    将 Aloha 格式的训练动作转换为 pi 内部格式，
    这样可以在 pi 的空间中训练模型。
    
    与 _encode_actions 的区别：
    - _encode_actions: pi -> Aloha（推理时使用）
    - _encode_actions_inv: Aloha -> pi（训练时使用）
    
    处理步骤（如果 adapt_to_pi=True）：
    1. 翻转某些关节的符号
    2. 反向转换夹爪（Aloha 归一化 -> pi 角度）
    
    Args:
        actions: [序列长度, 14] 动作序列，Aloha 格式
        adapt_to_pi: 是否执行空间转换
        
    Returns:
        转换后的动作序列，pi 内部格式
    """
    if adapt_to_pi:
        # 1. 翻转关节符号
        actions = _joint_flip_mask() * actions
        
        # 2. 反向转换夹爪角度
        # 使用逆函数将 Aloha 归一化角度转回 pi 角度
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
