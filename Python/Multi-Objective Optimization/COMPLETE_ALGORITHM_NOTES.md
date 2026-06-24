# 从教学版走向完整多目标算法

当前 Python 示例以教学拆解为主。它们故意写得短，是为了让新手看清 Pareto、非支配排序、拥挤距离、权重分解等核心概念。真正完整的算法还需要更多工程细节。

## 教学版和完整算法的区别

| 方面 | 教学版 | 完整版 |
| :--- | :--- | :--- |
| 目标 | 讲清概念 | 解决更复杂问题 |
| 数据规模 | 很小 | 可扩展 |
| 参数 | 少量固定参数 | 完整参数配置 |
| 约束处理 | 通常简化 | 需要明确约束策略 |
| 性能 | 可读性优先 | 需要效率优化 |
| 可视化 | 简单输出 | 曲线、前沿、指标记录 |

## 完整 NSGA-II 需要补齐什么

在 [mini_nsga2.py](examples/mini_nsga2.py) 基础上，完整版本通常还需要：

- 更清晰的个体结构。
- 可配置的变量上下界。
- 约束处理。
- 更完整的交叉和变异算子。
- 每代 Pareto 前沿记录。
- 指标统计，例如收敛曲线或超体积。
- 随机种子和重复实验。

## 完整 MOEA/D 需要补齐什么

在 [simple_moead_decomposition.py](examples/simple_moead_decomposition.py) 基础上，完整版本通常还需要：

- 系统生成权重向量。
- 邻域大小参数。
- 多种标量化方法。
- 理想点更新。
- 交叉和变异算子。
- 邻域替换策略。
- 结果保存和可视化。

## SPEA2 的学习入口

本目录新增了 [spea2_strength_demo.py](examples/spea2_strength_demo.py)，先演示 SPEA2 中的 strength 和 raw fitness 直觉。

SPEA2 完整版本还需要：

- 外部档案。
- 密度估计。
- 环境选择。
- 档案截断。

## Pareto 前沿可视化

本目录新增了 [pareto_front_ascii_plot.py](examples/pareto_front_ascii_plot.py)，用 ASCII 图显示非支配解位置。

如果后续使用 Matplotlib，可以把点和 Pareto 前沿画成散点图：

```python
import matplotlib.pyplot as plt

plt.scatter(xs, ys)
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.show()
```

## 与 Matlab 完整案例对照

如果想看更完整的算法资料，可以对照：

- [Matlab 多目标优化算法集合](<../../Matlab/Multi-Objective Optimization/README.md>)
- [Matlab 多目标优化算法索引](<../../Matlab/Multi-Objective Optimization/ALGORITHM_INDEX.md>)

推荐学习方式：

1. 先运行 Python 教学版。
2. 画出输入、目标值和 Pareto 前沿。
3. 再看 Matlab 完整实现。
4. 对照找出完整实现多出的工程细节。

## 下一步

先不要急着把所有算法都完整复刻。建议先选一个方向：NSGA-II、MOEA/D 或 SPEA2，把教学版扩展成“可配置参数 + 保存结果 + 可视化”的小项目。
