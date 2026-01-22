# Aloha Sim Docker 运行指南

本文档详细介绍如何使用 Docker 运行 OpenPI 的 Aloha Sim 仿真环境。

## 目录

- [前置条件](#前置条件)
- [环境准备](#环境准备)
- [NVIDIA Container Toolkit 安装](#nvidia-container-toolkit-安装)
- [代码准备](#代码准备)
- [模型配置说明](#模型配置说明)
- [运行仿真](#运行仿真)
- [架构说明](#架构说明)
- [常见问题](#常见问题)
- [验证和调试](#验证和调试)

## 前置条件

### 硬件要求
- **GPU**: NVIDIA GPU（推荐用于加速推理）
- **内存**: 至少 16GB RAM
- **存储**: 至少 20GB 可用空间（用于 Docker 镜像和模型文件）

### 软件要求
- **操作系统**: Linux (Ubuntu 22.04 推荐)
- **Docker**: 版本 20.10 或更高
- **Docker Compose**: 版本 2.0 或更高
- **NVIDIA 驱动**: 如果使用 GPU，需要安装 NVIDIA 驱动（版本 ≥ 525）

### 检查前置条件

```bash
# 检查 Docker 版本
docker --version

# 检查 Docker Compose 版本
docker compose version

# 检查 NVIDIA GPU（如果有）
nvidia-smi

# 检查 NVIDIA 驱动版本
nvidia-smi | grep "Driver Version"
```

## 环境准备

### 1. 克隆项目（如果尚未克隆）

```bash
git clone https://github.com/your-repo/openpi.git
cd openpi
```

### 2. 验证目录结构

确保以下文件存在：

```
openpi/
├── examples/
│   └── aloha_sim/
│       ├── compose.yml          # Docker Compose 配置
│       ├── Dockerfile           # 仿真环境 Dockerfile
│       ├── main.py              # 仿真主程序
│       ├── requirements.txt     # Python 依赖
│       └── README.md            # 说明文档
├── scripts/
│   └── serve_policy.py          # 策略服务器脚本
├── src/
│   └── openpi/                  # OpenPI 源代码
└── docker/
    └── serve_policy.Dockerfile  # 策略服务器 Dockerfile
```

## NVIDIA Container Toolkit 安装

如果您的系统有 NVIDIA GPU 并希望在容器中使用 GPU 加速，需要安装 NVIDIA Container Toolkit。

### 自动安装脚本

```bash
# 1. 配置软件源
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# 2. 更新并安装工具包
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. 配置 Docker 运行时
sudo nvidia-ctk runtime configure --runtime=docker

# 4. 重启 Docker 服务
sudo systemctl restart docker
```

### 验证安装

```bash
# 测试 GPU 访问
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

如果看到 GPU 信息输出，说明安装成功。

## 代码准备

### 已知问题修复

在运行前，需要修复代码中的语法错误（如果您的版本存在这些问题）。

#### 问题 1: 中文全角括号导致的语法错误

**文件**: `src/openpi/training/config.py`  
**位置**: 约第 117 行

**错误信息**:
```
SyntaxError: invalid character '（' (U+FF08)
```

**修复**: 将文档字符串中的中文全角括号 `（）` 改为半角括号 `()`，或者将该行移到文档字符串内部。

#### 问题 2: 重复的类定义

**文件**: `src/openpi/training/config.py`  
**位置**: 约第 248 行

**错误信息**:
```
IndentationError: expected an indented block after class definition on line 248
```

**修复**: 删除重复的 `@dataclasses.dataclass(frozen=True)` 装饰器和类定义。

确保文件中只有一个 `class DataConfigFactory(abc.ABC):` 定义。

### 验证修复

```bash
# 检查 Python 语法
python3 -m py_compile src/openpi/training/config.py

# 如果没有输出，说明语法正确
echo $?  # 应该输出 0
```

## 模型配置说明

### 环境变量配置

策略服务器通过环境变量 `SERVER_ARGS` 来配置运行参数。这些参数控制着使用哪个模型、环境配置以及推理参数。

#### 基本环境变量

```bash
# 指定环境类型（必需）
export SERVER_ARGS="--env ALOHA_SIM"

# 完整配置示例
export SERVER_ARGS="--env ALOHA_SIM --checkpoint pi0_aloha_sim --port 8000 --num_steps 10"
```

#### 可用参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--env` | 环境类型标识符 | 必需 | `ALOHA_SIM`, `ALOHA_REAL`, `UR5` |
| `--checkpoint` | 模型检查点名称 | 根据环境自动选择 | `pi0_aloha_sim` |
| `--port` | HTTP 服务端口 | `8000` | `8000`, `8080` |
| `--host` | 绑定的主机地址 | `0.0.0.0` | `0.0.0.0`, `127.0.0.1` |
| `--num_steps` | 扩散采样步数 | `10` | `5`, `10`, `20` |
| `--dtype` | 模型精度 | `bfloat16` | `float32`, `bfloat16` |

### 模型检查点详解

#### 检查点文件结构

模型检查点包含训练好的权重文件，存储在 Google Cloud Storage 上：

```
gs://openpi-assets/checkpoints/
├── pi0_aloha_sim/          # Aloha 仿真环境模型 (~11.2GB)
│   ├── checkpoint.pth       # 模型权重
│   ├── config.json          # 模型配置
│   └── norm_stats.json      # 归一化统计信息
├── pi0_aloha_real/         # Aloha 真实机器人模型
├── pi0_ur5/                # UR5 机器人模型
└── pi0_libero/             # Libero 仿真模型
```

#### 本地缓存位置

下载的模型会缓存到 Docker 卷中：

```bash
# Docker 卷名称
openpi_assets

# 容器内路径
/openpi_assets/openpi-assets/checkpoints/

# 查看卷内容
docker run --rm -v openpi_assets:/data alpine ls -lh /data/openpi-assets/checkpoints/
```

#### 检查点下载流程

1. **首次启动**：服务器检测到检查点不存在
2. **自动下载**：从 Google Cloud Storage 下载（约 11.2GB）
3. **进度显示**：
   ```
   INFO:openpi.shared.download:Downloading gs://openpi-assets/checkpoints/pi0_aloha_sim
   4% 472M/11.2G [01:06<25:23, 7.56MiB/s]
   ```
4. **加载模型**：下载完成后自动加载到内存
5. **后续启动**：直接使用缓存的检查点

### 模型配置文件

#### PI0 模型架构配置

模型的架构参数定义在 `src/openpi/models/pi0_config.py` 中：

```python
@dataclasses.dataclass(frozen=True)
class PI0Config:
    """PI0 模型配置
    
    Attributes:
        pi05: 是否使用 PI0.5 架构（使用 adaRMS 归一化）
        paligemma_variant: PaliGemma 子模型配置名称
        action_expert_variant: 动作专家子模型配置名称
        action_horizon: 动作序列长度（预测未来多少步）
        action_dim: 单步动作维度（通常为 32 维）
        dtype: 模型计算精度
    """
    pi05: bool = False
    paligemma_variant: str = "paligemma_3b"
    action_expert_variant: str = "gemma_2b"
    action_horizon: int = 50
    action_dim: int = 32
    dtype: str = "bfloat16"
```

#### 环境配置文件

每个环境有对应的配置文件，定义了：

```python
# examples/aloha_sim/env.py
class AlohaSimConfig:
    """Aloha 仿真环境配置"""
    
    # 模型配置
    checkpoint_name: str = "pi0_aloha_sim"
    action_horizon: int = 50
    action_dim: int = 14  # 7个关节 × 2个手臂
    
    # 观测配置
    image_size: tuple[int, int] = (480, 640)
    camera_names: list[str] = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    
    # 控制配置
    control_frequency: float = 50.0  # Hz
    max_episode_steps: int = 400
```

### 推理参数配置

#### 扩散采样步数 (num_steps)

控制从噪声生成动作的质量和速度：

```bash
# 快速推理（较低质量）
export SERVER_ARGS="--env ALOHA_SIM --num_steps 5"

# 标准推理（平衡质量和速度）
export SERVER_ARGS="--env ALOHA_SIM --num_steps 10"

# 高质量推理（较慢）
export SERVER_ARGS="--env ALOHA_SIM --num_steps 20"
```

**性能对比**：

| num_steps | 推理时间 | 动作质量 | 适用场景 |
|-----------|---------|---------|---------|
| 5 | ~50ms | 中等 | 快速原型、调试 |
| 10 | ~100ms | 良好 | 标准使用 |
| 20 | ~200ms | 优秀 | 高精度任务 |

#### 数据类型 (dtype)

影响计算精度和速度：

```bash
# 高精度（较慢，占用更多显存）
export SERVER_ARGS="--env ALOHA_SIM --dtype float32"

# 混合精度（推荐，平衡精度和性能）
export SERVER_ARGS="--env ALOHA_SIM --dtype bfloat16"
```

**对比**：

| dtype | 精度 | 速度 | 显存占用 | 推荐场景 |
|-------|------|------|---------|---------|
| float32 | 高 | 慢 | 大 | 调试、验证 |
| bfloat16 | 中 | 快 | 小 | 生产部署 |

### 自定义模型配置

#### 使用本地检查点

如果您有本地训练的模型检查点：

```yaml
# compose.yml 修改
services:
  openpi_server:
    volumes:
      - ./:/app
      - ./local_checkpoints:/local_checkpoints  # 挂载本地检查点目录
    environment:
      - SERVER_ARGS=--env ALOHA_SIM --checkpoint /local_checkpoints/my_model
```

#### 修改模型配置

创建自定义环境配置：

```python
# examples/aloha_sim/custom_env.py
from openpi.models.pi0_config import PI0Config

# 自定义 PI0.5 配置
custom_config = PI0Config(
    pi05=True,  # 使用 PI0.5 架构
    paligemma_variant="paligemma_3b",
    action_expert_variant="gemma_2b",
    action_horizon=100,  # 预测更长的动作序列
    action_dim=32,
    dtype="bfloat16"
)
```

### 模型加载流程详解

#### 启动顺序

1. **容器启动**：Docker Compose 创建并启动容器
2. **环境初始化**：设置 Python 环境和依赖
3. **解析参数**：读取 `SERVER_ARGS` 环境变量
4. **检查检查点**：
   ```python
   # 伪代码
   checkpoint_path = get_checkpoint_path(args.checkpoint)
   if not checkpoint_path.exists():
       download_checkpoint(args.checkpoint)
   ```
5. **下载模型**（如需要）：
   - 显示下载进度
   - 验证文件完整性
   - 解压和缓存
6. **加载模型**：
   ```python
   # 创建模型实例
   model = PI0Pytorch(config)
   
   # 加载权重
   checkpoint = torch.load(checkpoint_path)
   model.load_state_dict(checkpoint['model_state_dict'])
   
   # 移动到 GPU
   model = model.to(device)
   model.eval()
   ```
7. **编译优化**：
   ```python
   # PyTorch 2.0 编译优化
   model.sample_actions = torch.compile(
       model.sample_actions, 
       mode="max-autotune"
   )
   ```
8. **启动 HTTP 服务**：监听指定端口接受请求

#### 日志输出示例

```
openpi_server-1  | INFO:openpi.serving:Starting policy server
openpi_server-1  | INFO:openpi.serving:Environment: ALOHA_SIM
openpi_server-1  | INFO:openpi.serving:Checkpoint: pi0_aloha_sim
openpi_server-1  | INFO:openpi.shared.download:Downloading gs://openpi-assets/checkpoints/pi0_aloha_sim
openpi_server-1  | 100% 11.2G/11.2G [25:00<00:00, 7.56MiB/s]
openpi_server-1  | INFO:openpi.serving:Loading model checkpoint...
openpi_server-1  | INFO:openpi.serving:Model device: cuda:0
openpi_server-1  | INFO:openpi.serving:Model dtype: torch.bfloat16
openpi_server-1  | INFO:openpi.serving:Compiling model for optimization...
openpi_server-1  | INFO:openpi.serving:Model loaded successfully
openpi_server-1  | INFO:uvicorn:Started server process [1]
openpi_server-1  | INFO:uvicorn:Waiting for application startup.
openpi_server-1  | INFO:uvicorn:Application startup complete.
openpi_server-1  | INFO:uvicorn:Uvicorn running on http://0.0.0.0:8000
```

### 多模型配置

如果需要同时运行多个模型实例：

```yaml
# compose.yml
services:
  openpi_server_sim:
    image: openpi_server
    environment:
      - SERVER_ARGS=--env ALOHA_SIM --port 8000
    ports:
      - "8000:8000"
    
  openpi_server_real:
    image: openpi_server
    environment:
      - SERVER_ARGS=--env ALOHA_REAL --port 8001
    ports:
      - "8001:8001"
```

### 配置调优建议

#### 开发环境

```bash
# 优先速度，牺牲部分质量
export SERVER_ARGS="--env ALOHA_SIM --num_steps 5 --dtype bfloat16"
```

#### 生产环境

```bash
# 平衡质量和性能
export SERVER_ARGS="--env ALOHA_SIM --num_steps 10 --dtype bfloat16"
```

#### 演示/评估环境

```bash
# 最高质量
export SERVER_ARGS="--env ALOHA_SIM --num_steps 20 --dtype float32"
```

## 运行仿真

### 快速启动

使用 Docker Compose 一键启动整个仿真环境：

```bash
# 设置环境变量指定仿真环境
export SERVER_ARGS="--env ALOHA_SIM"

# 构建并启动容器
docker compose -f examples/aloha_sim/compose.yml up --build
```

### 分步说明

#### 1. 环境变量配置

```bash
export SERVER_ARGS="--env ALOHA_SIM"
```

这个环境变量告诉策略服务器使用 Aloha Sim 环境的配置。

#### 2. 构建镜像

首次运行或代码更改后需要重新构建：

```bash
docker compose -f examples/aloha_sim/compose.yml build
```

这将构建两个镜像：
- `openpi_server`: 策略服务器（包含 PI0 模型）
- `aloha_sim`: 仿真环境（MuJoCo + 环境交互）

#### 3. 启动服务

```bash
docker compose -f examples/aloha_sim/compose.yml up
```

或者在后台运行：

```bash
docker compose -f examples/aloha_sim/compose.yml up -d
```

#### 4. 查看日志

```bash
# 查看所有容器日志
docker compose -f examples/aloha_sim/compose.yml logs -f

# 只查看策略服务器日志
docker logs -f aloha_sim-openpi_server-1

# 只查看仿真环境日志
docker logs -f aloha_sim-runtime-1
```

#### 5. 停止服务

```bash
# 优雅停止
docker compose -f examples/aloha_sim/compose.yml down

# 强制停止并删除数据卷
docker compose -f examples/aloha_sim/compose.yml down -v
```

## 架构说明

### 容器组成

整个系统由两个 Docker 容器组成：

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                           │
│                                                              │
│  ┌──────────────────────────┐    ┌─────────────────────┐   │
│  │   openpi_server-1        │    │   runtime-1         │   │
│  │  (策略服务器)              │◄───│  (仿真环境)          │   │
│  │                          │    │                     │   │
│  │  - PI0 模型              │    │  - MuJoCo 仿真      │   │
│  │  - GPU 推理              │    │  - 观察/动作        │   │
│  │  - HTTP API (8000)       │    │  - 客户端逻辑       │   │
│  │  - CUDA 12.2             │    │  - EGL 渲染         │   │
│  └──────────────────────────┘    └─────────────────────┘   │
│           ▲                                                  │
│           │ /openpi_assets (共享卷)                          │
│           └──────────────────────────────────────────────────│
│                         模型检查点 (11.2GB)                   │
└─────────────────────────────────────────────────────────────┘
```

### 通信流程

1. **启动阶段**:
   - `openpi_server-1` 启动，下载/加载模型检查点
   - `runtime-1` 启动，等待服务器就绪
   - `runtime-1` 定期检查服务器健康状态（每3秒）

2. **运行阶段**:
   - `runtime-1` 从 MuJoCo 获取观察数据
   - 通过 HTTP 发送观察数据到 `openpi_server-1`
   - `openpi_server-1` 推理并返回动作
   - `runtime-1` 将动作应用到仿真环境

3. **停止阶段**:
   - 接收到停止信号（Ctrl+C）
   - 两个容器优雅关闭
   - 保存必要的状态和日志

### 数据卷

```yaml
volumes:
  - ./:/app              # 源代码挂载（开发模式）
  - openpi_assets:/openpi_assets  # 模型检查点共享存储
```

### 网络配置

```yaml
networks:
  default:
    name: openpi_network
```

两个容器在同一网络中，可以通过服务名相互访问。

### 端口映射

```yaml
openpi_server:
  ports:
    - "8000:8000"  # 策略 API 端口
```

服务器 API 可以从宿主机通过 `http://localhost:8000` 访问。

## 常见问题

### 1. 模型下载缓慢

**现象**: 看到 `Still waiting for server...` 持续很长时间

**原因**: 模型检查点文件约 11.2GB，首次运行需要从 Google Cloud Storage 下载。

**查看进度**:
```bash
docker logs aloha_sim-openpi_server-1 2>&1 | grep -E "(Downloading|%)"
```

**解决方案**:
- 耐心等待（通常需要 10-30 分钟，取决于网络速度）
- 使用代理加速（如果在国内）
- 或者手动下载模型文件并放置到正确位置

**手动下载模型**:
```bash
# 创建目录
mkdir -p ~/.cache/openpi_assets/openpi-assets/checkpoints

# 使用 gsutil 或其他工具下载
# gs://openpi-assets/checkpoints/pi0_aloha_sim
```

### 2. GPU 不可用错误

**错误信息**:
```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

**原因**: NVIDIA Container Toolkit 未正确安装或配置。

**解决步骤**:
1. 确认 NVIDIA 驱动已安装：`nvidia-smi`
2. 重新安装 NVIDIA Container Toolkit（参见上面的安装步骤）
3. 检查 Docker 配置：`cat /etc/docker/daemon.json`
4. 重启 Docker：`sudo systemctl restart docker`
5. 测试 GPU 访问：`docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi`

### 3. EGL 渲染错误

**错误信息**:
```
libEGL warning: egl: failed to create dri2 screen
MESA: warning: Driver does not support the 0x7d67 PCI ID
```

**说明**: 这些是警告信息，通常不影响仿真运行。MuJoCo 可以在软件渲染模式下运行。

**如果仿真无法启动**:
```bash
# 安装 EGL 依赖
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev

# 重新构建镜像
docker compose -f examples/aloha_sim/compose.yml build --no-cache
```

### 4. 端口冲突

**错误信息**:
```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**解决方案**:
```bash
# 查找占用端口的进程
sudo lsof -i :8000

# 停止占用端口的进程，或修改 compose.yml 使用不同端口
```

### 5. 内存不足

**错误信息**:
```
Cannot allocate memory
```

**解决方案**:
```bash
# 检查系统内存
free -h

# 增加 Docker 内存限制（Docker Desktop）
# 或清理不用的镜像和容器
docker system prune -a
```

### 6. 代码更改未生效

**原因**: Docker 使用了缓存的镜像。

**解决方案**:
```bash
# 强制重新构建（不使用缓存）
docker compose -f examples/aloha_sim/compose.yml build --no-cache

# 或者重新创建容器
docker compose -f examples/aloha_sim/compose.yml up --build --force-recreate
```

## 验证和调试

### 检查容器状态

```bash
# 查看运行中的容器
docker compose -f examples/aloha_sim/compose.yml ps

# 预期输出：
# NAME                         STATUS              PORTS
# aloha_sim-openpi_server-1   Up 5 minutes       0.0.0.0:8000->8000/tcp
# aloha_sim-runtime-1          Up 5 minutes
```

### 测试策略服务器

```bash
# 检查服务器健康状态
curl http://localhost:8000/health

# 预期输出：
# {"status": "healthy"}
```

### 查看实时日志

```bash
# 查看所有日志
docker compose -f examples/aloha_sim/compose.yml logs -f

# 只看最近 100 行
docker compose -f examples/aloha_sim/compose.yml logs --tail=100
```

### 进入容器调试

```bash
# 进入策略服务器容器
docker exec -it aloha_sim-openpi_server-1 bash

# 进入仿真环境容器
docker exec -it aloha_sim-runtime-1 bash
```

### 预期输出示例

**服务器启动成功**:
```
openpi_server-1  | INFO:openpi.shared.download:Downloading gs://openpi-assets/checkpoints/pi0_aloha_sim to /openpi_assets/openpi-assets/checkpoints/pi0_aloha_sim
openpi_server-1  | 100% 11.2G/11.2G [25:00<00:00, 7.56MiB/s]
openpi_server-1  | INFO:openpi.serving:Loading model checkpoint...
openpi_server-1  | INFO:openpi.serving:Model loaded successfully
openpi_server-1  | INFO:uvicorn:Started server process
openpi_server-1  | INFO:uvicorn:Waiting for application startup.
openpi_server-1  | INFO:uvicorn:Application startup complete.
openpi_server-1  | INFO:uvicorn:Uvicorn running on http://0.0.0.0:8000
```

**仿真开始运行**:
```
runtime-1        | INFO:root:Still waiting for server...
runtime-1        | INFO:root:Server is ready!
runtime-1        | INFO:root:Starting episode 0
runtime-1        | INFO:root:Step 0: got action from policy
runtime-1        | INFO:root:Step 1: got action from policy
...
```

## 性能优化

### GPU 加速

确保策略服务器使用 GPU：

```bash
# 查看 GPU 使用情况
nvidia-smi -l 1  # 每秒刷新一次

# 在日志中应该看到 GPU 被使用
docker logs aloha_sim-openpi_server-1 2>&1 | grep -i "cuda\|gpu"
```

### 减少下载时间

```bash
# 持久化模型文件（使用命名卷）
# compose.yml 中已配置：
volumes:
  openpi_assets:
    name: openpi_assets

# 模型只需下载一次，后续启动会直接使用缓存
```

### 并行处理

```bash
# 如果需要运行多个实例，修改 compose.yml：
docker compose -f examples/aloha_sim/compose.yml up --scale runtime=3
```

## 开发模式

### 代码热重载

当前配置已挂载源代码目录：

```yaml
volumes:
  - ./:/app
```

修改代码后，重启容器即可生效：

```bash
docker compose -f examples/aloha_sim/compose.yml restart
```

### 调试模式

添加调试端口：

```yaml
# 在 compose.yml 中添加：
openpi_server:
  ports:
    - "8000:8000"
    - "5678:5678"  # Python 调试器端口
  environment:
    - DEBUG=1
```

## 生产部署建议

### 1. 使用预构建镜像

```bash
# 构建并推送到镜像仓库
docker compose -f examples/aloha_sim/compose.yml build
docker tag openpi_server your-registry/openpi_server:latest
docker push your-registry/openpi_server:latest
```

### 2. 资源限制

在 `compose.yml` 中添加资源限制：

```yaml
services:
  openpi_server:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. 健康检查

```yaml
services:
  openpi_server:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### 4. 日志管理

```yaml
services:
  openpi_server:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
```

## 故障排除清单

遇到问题时，按以下顺序检查：

- [ ] NVIDIA 驱动已安装：`nvidia-smi`
- [ ] NVIDIA Container Toolkit 已安装：`nvidia-ctk --version`
- [ ] Docker 服务正常运行：`systemctl status docker`
- [ ] GPU 在 Docker 中可用：`docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi`
- [ ] 端口 8000 未被占用：`lsof -i :8000`
- [ ] 代码语法正确：`python3 -m py_compile src/openpi/training/config.py`
- [ ] 足够的磁盘空间：`df -h`
- [ ] 足够的内存：`free -h`
- [ ] 网络连接正常：`ping -c 3 8.8.8.8`

## 参考资源

- [OpenPI GitHub 仓库](https://github.com/your-repo/openpi)
- [NVIDIA Container Toolkit 文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [MuJoCo 文档](https://mujoco.readthedocs.io/)

## 更新日志

- **2026-01-20**: 初始版本，包含完整的安装和运行指南
- 修复了代码中的语法错误
- 添加了详细的故障排除步骤
- 包含了架构说明和性能优化建议

---

如有问题，请提交 Issue 或查看项目文档。
