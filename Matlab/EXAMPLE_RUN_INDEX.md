# Matlab 示例运行索引

这份索引帮助你快速判断：进入哪个目录、先看哪个说明、可能从哪个文件开始运行。

如果你是第一次运行 Matlab 示例，先读：

- [Matlab 示例运行指南](RUNNING_EXAMPLES.md)

## 多目标优化算法

目录：

- [Multi-Objective Optimization](<Multi-Objective Optimization/README.md>)

| 算法目录 | 优先阅读 | 常见入口或核心文件 | 说明 |
| :--- | :--- | :--- | :--- |
| [MODA](<Multi-Objective Optimization/MODA/README.md>) | `README.md`、`MODA-USE.md` | `MODA.m` | 多目标蜻蜓算法 |
| [MODE](<Multi-Objective Optimization/MODE/README.md>) | `README.md`、`MODE-USE.md` | `MODE.m`、`MODEparam.m` | 差分进化多目标优化 |
| [MOEA-D](<Multi-Objective Optimization/MOEA-D/README.md>) | `README.md`、`MOEA-D-USE.md` | `main.m`、`moead.m` | 基于分解的多目标进化算法 |
| [MOGA-Case](<Multi-Objective Optimization/MOGA-Case/README.md>) | `README.md`、`MOGA-Case-USE.md` | `main.m` | 多目标遗传算法案例 |
| [MOGOA](<Multi-Objective Optimization/MOGOA/README.md>) | `README.md`、`MOGOA-USE.md` | `MOGOA.m` | 多目标蝗虫优化算法 |
| [MOGWO](<Multi-Objective Optimization/MOGWO/README.md>) | `README.md`、`MOGWO-USE.md` | `MOGWO.m` | 多目标灰狼优化算法 |
| [MOMVO](<Multi-Objective Optimization/MOMVO/README.md>) | `README.md`、`MOMVO-USE.md` | `MOMVO.m` | 多目标多元宇宙优化算法 |
| [MSSA](<Multi-Objective Optimization/MSSA/README.md>) | `README.md`、`MSSA-USE.md` | `MSSA.m` | 多目标樽海鞘群算法 |
| [NSDBO](<Multi-Objective Optimization/NSDBO/README.md>) | `README.md`、`NSDBO-USE.md` | 查看说明文档 | 非支配排序蜣螂算法 |
| [NSGA-II](<Multi-Objective Optimization/NSGA-II/README.md>) | `README.md`、`NSGA-II-USE.md`、`NSGA-II-DA.md` | 查看说明文档 | 非支配排序遗传算法 II |
| [NSGA-III](<Multi-Objective Optimization/NSGA-III/README.md>) | `README.md`、`NSGA-III-USE.md`、`NSGA-III-DA.md` | 查看说明文档 | 非支配排序遗传算法 III |
| [NSWOA](<Multi-Objective Optimization/NSWOA/README.md>) | `README.md`、`NSWOA-USE.md` | 查看说明文档 | 多目标鲸鱼优化算法 |
| [PESA-II](<Multi-Objective Optimization/PESA-II/README.md>) | `README.md`、`PESA-II-USE.md` | 查看说明文档 | 基于范围选择的进化多目标优化 |
| [SPEA2](<Multi-Objective Optimization/SPEA2/README.md>) | `README.md`、`SPEA2-USE.md` | 查看说明文档 | 强度 Pareto 进化算法 |

运行建议：

1. 进入算法目录。
2. 阅读 `README.md` 和 `*-USE.md`。
3. 找到主脚本或说明文档指定入口。
4. 先用默认参数运行。
5. 再修改目标函数、种群数量、迭代次数等参数。

## 神经网络案例

目录：

- [Neural Network](<Neural Network/README.md>)

| 任务目录 | 模型目录 | 常见入口 | 适合场景 |
| :--- | :--- | :--- | :--- |
| [Classification](<Neural Network/Classification/README.md>) | BP | `Main.m` | 基础分类任务 |
| [Classification](<Neural Network/Classification/README.md>) | CNN | `main.m` | 图像或局部特征分类 |
| [Classification](<Neural Network/Classification/README.md>) | ELM | `main.m` | 快速分类实验 |
| [Classification](<Neural Network/Classification/README.md>) | GA-BP | `main.m` | 遗传算法优化 BP 分类 |
| [Classification](<Neural Network/Classification/README.md>) | LSTM | `main.m` | 序列分类 |
| [Classification](<Neural Network/Classification/README.md>) | PSO-BP | `main.m` | 粒子群优化 BP 分类 |
| [Classification](<Neural Network/Classification/README.md>) | RBF | `main.m` | 径向基分类 |
| [Classification](<Neural Network/Classification/README.md>) | RF | `main.m` | 随机森林分类 |
| [Classification](<Neural Network/Classification/README.md>) | SVM | `main.m` | 支持向量机分类 |
| [Regression](<Neural Network/Regression/README.md>) | BP / CNN / ELM / GA-BP / LSTM / PSO-BP / RBF / RF / SVM | 多数为 `main.m` | 连续数值预测 |
| [Time Series Prediction](<Neural Network/Time Series Prediction/README.md>) | BP / CNN / ELM / GA-BP / LSTM / PSO-BP / RBF / RF / SVM | 多数为 `main.m` | 时间序列预测 |

运行建议：

1. 先看 [模型选择指南](<Neural Network/MODEL_SELECTION.md>)。
2. 明确任务是分类、回归还是时间序列预测。
3. 进入具体模型目录。
4. 检查数据文件和工具箱。
5. 先跑默认示例，再替换自己的数据。

## 结果记录模板

建议把每次实验记录成下面的格式：

```markdown
## 实验记录

- 日期：
- 目录：
- 入口文件：
- 数据文件：
- 关键参数：
- 输出结果：
- 是否复现：
- 遇到的问题：
```

