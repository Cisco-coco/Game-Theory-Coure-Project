# 运行环境

项目以 Python 3.10 和 OpenSpiel 1.6.11 为当前验证基准，直接依赖见 `requirements.txt`。它是直接依赖版本清单，不是跨操作系统的完整锁文件。测试使用标准库 unittest，无额外测试依赖。

## 本机

使用已有 Git Bash 和 Anaconda `tor`。已验证 Python 3.10.14、OpenSpiel 1.6.11+local.mingw、NumPy 1.26.4、Matplotlib 3.9.2、PyTorch 2.4.1；CPU 与 CUDA 路径均有测试覆盖。不需要重新安装或替换共享环境中的包。

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tor
python -c 'import pyspiel, torch; print(pyspiel.load_game("leduc_poker")); print(torch.__version__, torch.cuda.is_available())'
```

Conda 未进入 PATH 不表示没有安装；可直接使用已有环境的 Python 可执行文件。`pokerkit_wrapper` 的可选导入提示不影响本项目 Kuhn/Leduc，勿为消除该提示无必要地安装额外包。

## 其他机器

先检查现有 Python 和依赖。只有缺失或存在兼容性冲突时才补装；确认平台支持后使用 `python -m pip install -r requirements.txt`。这不保证所有 Windows/Python 组合都存在 OpenSpiel 二进制包。PyTorch 的 CPU/CUDA 构建也应按目标硬件选择。

本机 OpenSpiel 为 1.6.11 源码的 MinGW 本地构建，版本后缀为 `+local.mingw`，不是可从公共包索引直接下载的官方 Windows wheel。该后缀满足 `open-spiel==1.6.11` 的公共版本要求。本地构建补齐 JSON CMake 模板、链接 Python 库、附带 MinGW 运行时 DLL；未修改游戏或求解算法。ACPC 等可选扩展未启用。

本机 wheel、构建源码与构建记录独立保存，不提交到此仓库。若目标机器无法取得兼容包，需要单独准备兼容构建；不要假定将版本号改为另一版本就能精确复现实验。

所有训练入口记录实际依赖版本和相关源码哈希。升级依赖后先运行完整测试，再重新建立相应基线。
