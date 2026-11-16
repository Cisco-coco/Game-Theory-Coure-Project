# CFR Poker AI

双人零和扑克求解项目，支持 OpenSpiel 的 Kuhn Poker 和 Leduc Poker。实现 CFR、CFR+、External-sampling MCCFR 和 PyTorch Deep CFR，使用精确 NashConv / 2 评估平均策略。

## 环境

优先使用已有 Git Bash 与 Conda `tor`，无需重新安装本机依赖：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tor
python -m unittest discover -s tests -p 'test_*.py' -v
```

当前验证版本：Python 3.10.14、OpenSpiel 1.6.11+local.mingw、NumPy 1.26.4、Matplotlib 3.9.2、PyTorch 2.4.1。依赖及跨机器安装注意事项见 [环境说明](docs/environment.md)。测试统一使用标准库 unittest。

## 运行

在仓库根目录执行；输出目录必须为空或不存在。

```bash
# 本地 CFR/CFR+ 与 OpenSpiel 参考实现对照
python compare_tabular.py --game leduc_poker --iterations 300 --eval-every 20 --output results/tabular

# 采样求解器；可选地读取同环境、同游戏的表格基线
python train_mccfr.py --game leduc_poker --iterations 50000 --eval-every 2500 --baseline results/tabular --output results/mccfr

# Deep CFR：每轮训练平均策略，评估频率不改变训练日程
python train_deep_cfr.py --game leduc_poker --iterations 100 --eval-every 10 --hidden 128 --traversals 400 --capacity 50000 --advantage-steps 500 --strategy-steps 1000 --output results/deep

# 宽度64/128的五种子串行对比
bash run_width_study.sh results/width-study
```

Deep CFR 默认使用 CPU 单线程。可指定 `--device cuda`；小网络不保证 GPU 更快。快速试运行可减少 `--iterations`、`--traversals` 和训练步数；完整命令参数见各入口的 `--help`。

## 代码结构

| 文件 | 职责 |
|---|---|
| `tabular_cfr.py` | 完整遍历 CFR/CFR+ |
| `external_mccfr.py` | 外部采样 MCCFR、稀疏信息集表 |
| `deep_cfr.py` | 优势/平均策略网络、采样训练、完整检查点 |
| `reservoir.py` | Algorithm R 蓄水池采样 |
| `evaluation.py` | 精确可利用度和自对弈收益 |
| `experiment_io.py` | CSV/JSON 写入 |
| `compare_tabular.py` | 表格求解器与官方实现对照 |
| `train_mccfr.py` / `train_deep_cfr.py` | 多种子训练、指标、策略导出及绘图 |
| `tests/` | 差分验证、策略不变量、采样与恢复测试 |

## 产物与恢复

训练输出 CSV、JSON、曲线图及 Deep CFR 模型，统一放在被 Git 忽略的 `results/`。仓库不包含训练权重、历史实验产物或实验报告。

```bash
python train_deep_cfr.py --resume results/deep/seed-7/checkpoint.pt --iterations 200 --output results/deep-resumed
```

恢复使用检查点中的配置、种子和评估间隔；`--iterations` 是总目标轮数，必须大于已完成轮数。每个种子默认仅保留最新完整模型；加上 `--keep-checkpoints` 可保留编号快照。详细契约见 [运行约定](docs/running.md)。

## 范围与引用

只支持双人 Kuhn/Leduc；没有扩展扑克环境。可利用度计算需要遍历完整游戏树，不适合直接套用到大规模扑克。不同算法每轮工作量不同，不能仅按轮数比较效率；跨种子标准差不是置信区间。

求解器参考并改编 OpenSpiel，保留上游版权与 Apache-2.0 许可说明；实现差异和论文引用见 [THIRD_PARTY.md](THIRD_PARTY.md)。项目实现不应标注为原论文完整复现或原创算法。
