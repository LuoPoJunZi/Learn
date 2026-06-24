# Python 多目标优化算法索引

这个索引用来整理当前 Python 目录下已经提供的多目标优化示例，以及后续可以扩展的算法方向。

## 当前示例

| 示例 | 文件 | 核心概念 | 难度 |
| :--- | :--- | :--- | :--- |
| 加权和网格搜索 | [examples/weighted_sum_grid_search.py](examples/weighted_sum_grid_search.py) | 权重偏好、目标合成 | 入门 |
| 简化 Pareto 筛选 | [examples/simple_pareto_front.py](examples/simple_pareto_front.py) | 支配关系、非支配解 | 入门 |
| Pareto 前沿 ASCII 可视化 | [examples/pareto_front_ascii_plot.py](examples/pareto_front_ascii_plot.py) | 前沿位置、可视化直觉 | 入门 |
| 迷你 NSGA-II | [examples/mini_nsga2.py](examples/mini_nsga2.py) | 非支配排序、拥挤距离、交叉变异 | 进阶 |
| 简化 MOEA/D | [examples/simple_moead_decomposition.py](examples/simple_moead_decomposition.py) | 权重向量、分解思想、邻域更新 | 进阶 |
| SPEA2 strength 演示 | [examples/spea2_strength_demo.py](examples/spea2_strength_demo.py) | 强度值、原始适应度 | 进阶 |

## 怎么选示例

如果你完全没接触过多目标优化：

1. 先看加权和网格搜索。
2. 再看 Pareto 筛选。
3. 再看 NSGA-II。
4. 最后看 MOEA/D。

如果你已经看过 Matlab 目录中的多目标算法：

- 想理解 NSGA-II 流程：看 [mini_nsga2.py](examples/mini_nsga2.py)。
- 想理解 MOEA/D 分解思想：看 [simple_moead_decomposition.py](examples/simple_moead_decomposition.py)。
- 想理解 Pareto 前沿：看 [simple_pareto_front.py](examples/simple_pareto_front.py)。

## 与 Matlab 目录的关系

Matlab 目录保存的是更完整的算法资料和案例：

- [Matlab 多目标优化算法集合](<../../Matlab/Multi-Objective Optimization/README.md>)
- [Matlab 示例运行索引](../../Matlab/EXAMPLE_RUN_INDEX.md)

Python 目录当前更偏“教学拆解版”，适合先理解算法骨架。理解后再看 Matlab 完整实现，会轻松很多。

## 已补充的扩展入口

- 完整 NSGA-II / MOEA/D 扩展说明：[COMPLETE_ALGORITHM_NOTES.md](COMPLETE_ALGORITHM_NOTES.md)
- SPEA2 基础直觉：[examples/spea2_strength_demo.py](examples/spea2_strength_demo.py)
- Pareto 前沿可视化：[examples/pareto_front_ascii_plot.py](examples/pareto_front_ascii_plot.py)
- Matlab 完整算法对照：[Matlab 多目标优化算法集合](<../../Matlab/Multi-Objective Optimization/README.md>)

PESA-II、多目标粒子群、多目标灰狼、多目标鲸鱼等完整算法，当前优先通过 Matlab 目录对照学习；Python 目录保持教学拆解定位，避免把入门目录一次性变成大型算法库。
