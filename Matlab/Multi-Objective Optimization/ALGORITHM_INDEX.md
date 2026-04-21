# 多目标优化算法索引

这个索引用来帮助你快速选择和查找多目标优化算法。

## 什么是多目标优化

多目标优化用于同时优化多个目标，而这些目标之间往往互相冲突。

例如：

- 成本越低越好，但性能越高越好。
- 路径越短越好，但风险越低越好。
- 模型误差越小越好，但复杂度越低越好。

多目标优化通常不会只得到一个最优解，而是得到一组折中解，也就是 Pareto 前沿。

## 算法总览

| 缩写 | 全称 | 中文说明 | 类型 | 目录 |
| :--- | :--- | :--- | :--- | :--- |
| MODA | Multi-Objective Dragonfly Algorithm | 多目标蜻蜓算法 | 群智能 | [MODA](MODA/) |
| MODE | Multi-Objective Differential Evolution | 差分进化多目标优化算法 | 进化算法 | [MODE](MODE/) |
| MOEA-D | Multi-Objective Evolutionary Algorithm Based on Decomposition | 基于分解的多目标进化算法 | 分解方法 | [MOEA-D](MOEA-D/) |
| MOGA | Multi-Objective Genetic Algorithm Case | 多目标遗传算法案例 | 遗传算法 | [MOGA-Case](MOGA-Case/) |
| MOGOA | Multi-Objective Grasshopper Optimization Algorithm | 多目标蝗虫优化算法 | 群智能 | [MOGOA](MOGOA/) |
| MOGWO | Multi-Objective Grey Wolf Optimization Algorithm | 多目标灰狼优化算法 | 群智能 | [MOGWO](MOGWO/) |
| MOMVO | Multi-Objective Multi-Verse Optimization | 多目标多元宇宙优化算法 | 群智能 | [MOMVO](MOMVO/) |
| MSSA | Multi-Objective Salp Swarm Algorithm | 多目标樽海鞘群算法 | 群智能 | [MSSA](MSSA/) |
| NSDBO | Nondominated Sorting Beetle Algorithm | 非支配排序蜣螂算法 | 群智能 / 非支配排序 | [NSDBO](NSDBO/) |
| NSGA-II | Nondominated Sorting Genetic Algorithm II | 非支配排序遗传算法 II | 经典多目标进化算法 | [NSGA-II](NSGA-II/) |
| NSGA-III | Nondominated Sorting Genetic Algorithm III | 非支配排序遗传算法 III | 高维多目标进化算法 | [NSGA-III](NSGA-III/) |
| NSWOA | Multi-Objective Whale Optimization Algorithm | 多目标鲸鱼优化算法 | 群智能 | [NSWOA](NSWOA/) |
| PESA-II | Pareto Envelope-based Selection Algorithm II | 基于 Pareto 包络选择的算法 | Pareto 选择 | [PESA-II](PESA-II/) |
| SPEA2 | Strength Pareto Evolutionary Algorithm 2 | 强度 Pareto 进化算法 | 经典多目标进化算法 | [SPEA2](SPEA2/) |

## 新手推荐学习顺序

1. [NSGA-II](NSGA-II/)：经典算法，适合理解非支配排序和拥挤距离。
2. [SPEA2](SPEA2/)：理解强度 Pareto 和外部档案机制。
3. [MOEA-D](MOEA-D/)：理解基于分解的多目标优化思想。
4. [NSGA-III](NSGA-III/)：理解高维目标场景。
5. 群智能算法：MODA、MOGWO、MOGOA、MOMVO、MSSA、NSWOA。

## 按需求选择

| 需求 | 建议优先看 |
| :--- | :--- |
| 学多目标优化基础 | NSGA-II |
| 想理解 Pareto 档案 | SPEA2、PESA-II |
| 目标数量较多 | NSGA-III |
| 想看分解思想 | MOEA-D |
| 想看群智能算法 | MOGWO、MODA、MOMVO、MSSA |
| 想快速跑案例 | MOGA-Case |

## 运行前检查

进入具体算法目录后，建议先找：

- `README.md`
- `*-USE.md`
- `main.m`
- 算法主文件，例如 `NSGAII.m`、`MODA.m`、`MODE.m`
- 测试函数，例如 `ZDT1.m`、`MOP2.m`

运行前确认当前 Matlab 工作目录是对应算法目录，避免找不到函数或数据文件。

## 建议统一的算法说明模板

以后新增或整理算法时，建议 README 包含：

```markdown
# 算法名称

## 算法简介

## 适用场景

## 文件说明

## 如何运行

## 主要参数

## 输出结果

## 参考来源
```

