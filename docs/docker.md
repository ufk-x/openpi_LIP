### Docker 设置

本仓库中的所有示例都提供了正常运行和使用 Docker 运行的说明。虽然不是必需的，但推荐使用 Docker 选项，因为这将简化软件安装，提供更稳定的环境，并且对于依赖 ROS 的示例，还可以避免在你的机器上安装 ROS 而造成混乱。

- 基本的 Docker 安装说明在[这里](https://docs.docker.com/engine/install/)。
- Docker 必须以[无根模式（rootless mode）](https://docs.docker.com/engine/security/rootless/)安装。
- 要使用 GPU，你还必须安装 [NVIDIA 容器工具包](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
- 使用 `snap` 安装的 Docker 版本与 NVIDIA 容器工具包不兼容，会导致无法访问 `libnvidia-ml.so`（[问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)）。可以使用 `sudo snap remove docker` 卸载 snap 版本。
- Docker Desktop 也与 NVIDIA 运行时不兼容（[问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)）。可以使用 `sudo apt remove docker-desktop` 卸载 Docker Desktop。


如果从头开始，且你的主机是 Ubuntu 22.04，可以使用便捷脚本 `scripts/docker/install_docker_ubuntu22.sh` 和 `scripts/docker/install_nvidia_container_toolkit.sh` 完成上述所有操作。

使用以下命令构建 Docker 镜像并启动容器：
```bash
docker compose -f scripts/docker/compose.yml up --build
```

要为特定示例构建和运行 Docker 镜像，使用以下命令：
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
其中 `<example_name>` 是你想要运行的示例名称。

在首次运行任何示例时，Docker 会构建镜像。这期间可以去喝杯咖啡休息一下。后续运行会更快，因为镜像已被缓存。