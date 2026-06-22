# Python 多目标优化示例

这个目录保存 Python 版多目标优化入门示例。它和 Matlab 目录下的多目标优化资料定位不同：这里先用更少的代码讲清楚概念，让新手理解 Pareto 最优、非支配解、目标函数冲突等基础，再逐步扩展到更完整的算法。

## 你会学到什么

- 什么是多目标优化
- 为什么多个目标之间会冲突
- 什么是 Pareto 最优解
- 什么是非支配排序
- 如何用 Python 表示候选解、目标值和 Pareto 前沿

## 示例列表

| 示例 | 文件 | 说明 | 依赖 |
| :--- | :--- | :--- | :--- |
| 加权和网格搜索 | [examples/weighted_sum_grid_search.py](examples/weighted_sum_grid_search.py) | 用不同权重把两个目标合成一个目标 | 标准库 |
| 简化 Pareto 筛选 | [examples/simple_pareto_front.py](examples/simple_pareto_front.py) | 从一批候选解中找非支配解 | 标准库 |
| 迷你 NSGA-II | [examples/mini_nsga2.py](examples/mini_nsga2.py) | 演示非支配排序、拥挤距离、选择、交叉和变异 | 标准库 |
| 简化 MOEA/D | [examples/simple_moead_decomposition.py](examples/simple_moead_decomposition.py) | 演示权重向量、分解思想和邻域更新 | 标准库 |

更完整的列表可以看 [算法索引](ALGORITHM_INDEX.md)。

## 如何运行

进入仓库根目录后运行：

```powershell
python "Python\Multi-Objective Optimization\examples\weighted_sum_grid_search.py"
python "Python\Multi-Objective Optimization\examples\simple_pareto_front.py"
python "Python\Multi-Objective Optimization\examples\mini_nsga2.py"
python "Python\Multi-Objective Optimization\examples\simple_moead_decomposition.py"
```

如果你的系统使用 `py` 命令：

```powershell
py "Python\Multi-Objective Optimization\examples\simple_pareto_front.py"
```

## 核心概念

### 单目标优化

单目标优化只有一个目标，例如：

```text
让成本最低
```

### 多目标优化

多目标优化同时考虑多个目标，例如：

```text
成本最低，同时质量最高
```

这两个目标可能互相冲突：质量越高，成本可能越高。

### Pareto 最优

如果一个解无法在“不让任何目标变差”的前提下继续改进，那么它就是 Pareto 最优解。

所有 Pareto 最优解组成的集合，通常称为 Pareto 前沿。

## 推荐学习顺序

1. 先运行加权和示例，理解“权重改变，偏好也会改变”。
2. 再运行 Pareto 筛选示例，理解“非支配解不是单个最优点，而是一组折中方案”。
3. 运行迷你 NSGA-II，理解“种群、排序、选择、交叉、变异”的整体流程。
4. 运行简化 MOEA/D，理解“把多目标问题拆成多个子问题”的思路。
5. 尝试修改候选解列表。
6. 尝试增加第三个目标。
7. 后续再学习完整 NSGA-II、MOEA/D、SPEA2 等算法。

## 迷你 NSGA-II 说明

[mini_nsga2.py](examples/mini_nsga2.py) 是为了学习写的简化版本，不追求工程完整性，但保留了 NSGA-II 的关键思想：

- 用一组候选解组成种群。
- 用非支配排序划分不同等级。
- 用拥挤距离保持解的多样性。
- 用锦标赛选择挑选父代。
- 用交叉和变异生成新解。

如果你已经看过 Matlab 里的 NSGA-II 示例，可以用这个 Python 版本对照理解流程。

## 简化 MOEA/D 说明

[simple_moead_decomposition.py](examples/simple_moead_decomposition.py) 演示的是 MOEA/D 的核心直觉：

- 不直接找一个“唯一最优解”。
- 用多个权重向量表示不同偏好。
- 每个权重向量对应一个标量子问题。
- 子问题之间共享邻域信息，逐步靠近 Pareto 前沿。

## 后续扩展方向

可以继续补充：

- Python 版 NSGA-II
- Python 版 MOEA/D
- 使用 NumPy 加速目标函数计算
- 使用 Matplotlib 绘制 Pareto 前沿
- 与 Matlab 算法结果对比
