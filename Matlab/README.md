# Matlab 学习与算法资料库

这个目录整理 Matlab 基础语法、工程计算示例、多目标优化算法和神经网络案例。

内容适合：

- 刚开始学习 Matlab 的新手
- 想查 Matlab 基础语法和常用操作的人
- 想运行多目标优化算法示例的人
- 想用 Matlab 做分类、回归、时间序列预测的人

## 推荐阅读路线

1. [Basics](Basics/README.md)：先学习 Matlab 基础语法、矩阵、绘图、脚本和并行计算。
2. [Multi-Objective Optimization](<Multi-Objective Optimization/README.md>)：再学习多目标优化算法。
3. [Neural Network](<Neural Network/README.md>)：最后按任务选择神经网络模型。

如果你已经有明确任务，可以直接看：

- [多目标优化算法索引](<Multi-Objective Optimization/ALGORITHM_INDEX.md>)
- [神经网络模型选择指南](<Neural Network/MODEL_SELECTION.md>)

## 目录说明

### Basics

[Basics](Basics/README.md) 保存 Matlab 基础学习笔记，包括：

- 基础命令
- 变量、函数、保留字
- 向量和矩阵
- 数组运算
- 绘图
- 脚本和函数
- 并行计算

### Multi-Objective Optimization

[Multi-Objective Optimization](<Multi-Objective Optimization/README.md>) 保存多目标优化算法代码和说明，例如：

- NSGA-II
- NSGA-III
- MOEA-D
- SPEA2
- PESA-II
- MODA
- MODE
- MOGWO
- MOGOA
- MOMVO

建议先看 [多目标优化算法索引](<Multi-Objective Optimization/ALGORITHM_INDEX.md>)，再进入具体算法目录。

### Neural Network

[Neural Network](<Neural Network/README.md>) 按任务类型组织：

- Classification：分类
- Regression：回归
- Time Series Prediction：时间序列预测

每个任务下包含 BP、CNN、ELM、GA-BP、LSTM、PSO-BP、RBF、RF、SVM 等模型案例。新手建议先看 [神经网络模型选择指南](<Neural Network/MODEL_SELECTION.md>)。

## 运行环境建议

建议使用较新的 Matlab 版本。不同案例可能需要不同工具箱：

- Deep Learning Toolbox：神经网络、CNN、LSTM 等深度学习相关案例
- Statistics and Machine Learning Toolbox：SVM、随机森林、统计建模相关案例
- Parallel Computing Toolbox：并行计算相关内容
- Optimization Toolbox：部分优化问题可能会用到

目录中包含 `.m`、`.mat`、`.xlsx`、图片和 `.mexw64` 文件。`.mexw64` 通常只适用于 Windows 64 位 Matlab 环境。

## 运行代码前的建议

1. 先阅读当前算法目录下的 `README.md` 或 `*-USE.md`。
2. 确认数据文件是否在代码期望的位置。
3. 不要随意移动 `.m`、`.mat`、`.xlsx` 文件，部分脚本可能依赖相对路径。
4. 如果运行报错，优先检查当前工作目录是否是案例所在文件夹。
5. 对包含随机初始化的算法，建议记录随机种子或多次运行取平均结果。

## 常见文件类型

| 类型 | 说明 |
| :--- | :--- |
| `.m` | Matlab 脚本或函数文件 |
| `.mat` | Matlab 数据文件 |
| `.xlsx` | Excel 数据集 |
| `.md` | 说明文档 |
| `.mexw64` | Windows 64 位编译扩展 |
| `.tif` / `.jpg` | 图像数据或结果图 |

## 维护建议

新增案例时，建议至少补充：

- 算法或模型简介
- 运行入口
- 数据文件说明
- 参数说明
- 输出结果说明
- 参考来源
