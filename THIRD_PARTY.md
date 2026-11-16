# 第三方来源与实现边界

## OpenSpiel

游戏规则、信息集与策略容器、精确最佳应对和期望收益计算均由 [OpenSpiel](https://github.com/google-deepmind/open_spiel) 提供。当前运行基准版本为 1.6.11；本机使用 1.6.11+local.mingw 构建，算法未作本地库修改。

上游版权为 DeepMind Technologies Limited；许可副本见 [Apache-2.0](licenses/OpenSpiel-Apache-2.0.txt)。改编文件保留版权、许可和修改说明。以下来源版本描述代码出处，并不表示需要安装多个运行版本。

| 本地文件 | 来源 | 改动 |
|---|---|---|
| `tabular_cfr.py` | OpenSpiel 2.0.2 的 `python/algorithms/cfr.py` | 数组存储；按被更新玩家返回标量价值；每遍历统一累加遗憾；同步/交替更新；独立策略快照和节点计数 |
| `external_mccfr.py` | OpenSpiel 1.6.11 的 `python/algorithms/external_sampling_mccfr.py`、`mccfr.py` | 独立 RNG；包含玩家编号的稀疏键；策略快照；训练计数 |
| `deep_cfr.py` | OpenSpiel 1.6.11 的 `python/pytorch/deep_cfr.py` | PyTorch 标准 MLP；合法动作屏蔽；独立随机流；冻结策略缓存；完整检查点和设备支持 |

源代码链接：
- [CFR](https://github.com/google-deepmind/open_spiel/blob/v2.0.2/open_spiel/python/algorithms/cfr.py)
- [MCCFR](https://github.com/google-deepmind/open_spiel/tree/v1.6.11/open_spiel/python/algorithms)
- [Deep CFR](https://github.com/google-deepmind/open_spiel/blob/v1.6.11/open_spiel/python/pytorch/deep_cfr.py)

本地求解器不调用官方求解器完成训练；对照入口和差分测试会调用官方实现。CFR+ 使用交替更新、非负遗憾截断和无延迟线性平均。MCCFR 保留双人 SIMPLE 平均与 1e-6 初始伪计数。

Deep CFR 每玩家采样后重置优势网络及优化器；保留历史蓄水池。优势全非正时按合法动作最大优势回退。损失先按每个样本的合法动作数归一化，再乘 t/T；平均策略网络每轮持续训练。上述初始化、损失和训练日程等选择与上游存在区别，不将本地实现等同于官方未修改版本或原论文完整复现。

## 项目代码与其他依赖

`reservoir.py` 为本项目编写的 Algorithm R 实现，参考 Vitter 的论文，没有复制论文代码；它用于 Deep CFR 的历史样本缓冲区。

训练入口、评估封装、产物写入、Shell 脚本及测试由本项目编写。NumPy、Matplotlib 和 PyTorch 通过包导入，没有复制其源码；其许可证以各安装包为准。

## 学术引用

1. Zinkevich et al. (2007). *Regret Minimization in Games with Incomplete Information*. [NeurIPS](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html).
2. Tammelin et al. (2015). *Solving Heads-up Limit Texas Hold’em*. [论文](https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf).
3. Lanctot et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games*. [NeurIPS](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html).
4. Brown et al. (2019). *Deep Counterfactual Regret Minimization*. [ICML](https://proceedings.mlr.press/v97/brown19b.html).
5. Vitter (1985). *Random Sampling with a Reservoir*. [DOI](https://doi.org/10.1145/3147.3165).
6. Lanctot et al. (2019). *OpenSpiel: A Framework for Reinforcement Learning in Games*. [arXiv](https://arxiv.org/abs/1908.09453).
