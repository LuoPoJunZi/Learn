# 时间序列预测神经网络示例 (Time Series Prediction)

## 适合人群与目标

适合已掌握 Matlab 矩阵和基本绘图、准备阅读现有模型案例的人。本目录的任务目标是预测未来时刻的数值，不是按模型名字判断哪个算法一定更好。

先阅读 [示例运行指南](../../RUNNING_EXAMPLES.md) 和 [现代深度学习工作流](../MODERN_WORKFLOW_GUIDE.md)，确认数据、工具箱与接口版本，再进入下列原始案例。

## 案例入口

| 模型 | 原有讲解 |
| :--- | :--- |
| BP | [BP/README.md](BP/README.md) |
| CNN | [CNN/README.md](CNN/README.md) |
| ELM | [ELM/README.md](ELM/README.md) |
| GA-BP | [GA-BP/README.md](GA-BP/README.md) |
| LSTM | [LSTM/README.md](LSTM/README.md) |
| PSO-BP | [PSO-BP/README.md](PSO-BP/README.md) |
| RBF | [RBF/README.md](RBF/README.md) |
| RF | [RF/README.md](RF/README.md) |
| SVM | [SVM/README.md](SVM/README.md) |

## 阅读与验证顺序

1. 先查案例输入数据和入口脚本，不直接把文件名改成自己的数据路径。
2. 核对时间顺序划分、输入窗口、预测步长与数据泄漏。
3. 用小样本运行，固定随机种子；先核对维度与输出，再调整超参数。
4. 比较模型时使用同一划分和指标，不把单次演示结果当作性能排名。

这些目录保留历史实现，不保证都使用最新接口。缺少工具箱时应先看版本与依赖说明，不要为绕过报错随意替换函数。

返回 [神经网络总入口](../README.md) 或 [Matlab 示例运行索引](../../EXAMPLE_RUN_INDEX.md)。
