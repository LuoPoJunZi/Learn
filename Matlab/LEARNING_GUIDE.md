# Matlab 学习导读

Matlab 很适合矩阵计算、工程仿真、绘图、优化算法和神经网络实验。新手学习时不要先陷进大型算法代码，先把“矩阵、脚本、函数、绘图、数据导入导出”打稳。

## 适合人群

- 刚开始学习 Matlab 的学生或工程用户。
- 想运行本目录多目标优化、神经网络案例的人。
- 会运行代码，但不清楚数据、路径、工具箱关系的人。

## 学习目标

学完后，你应该能：

- 理解矩阵和数组运算的区别。
- 编写脚本和函数。
- 导入 `.mat`、`.xlsx`、`.csv` 数据。
- 绘制并美化常用图形。
- 按任务选择优化算法或神经网络模型。

## 推荐阅读顺序

1. [Matlab 基础教程](Basics/README.md)
2. [Matlab 示例运行指南](RUNNING_EXAMPLES.md)
3. [Matlab 示例运行索引](EXAMPLE_RUN_INDEX.md)
4. [多目标优化算法索引](<Multi-Objective Optimization/ALGORITHM_INDEX.md>)
5. [神经网络模型选择指南](<Neural Network/MODEL_SELECTION.md>)

## 学习地图

```text
基础命令
-> 向量和矩阵
-> 数组运算
-> 脚本和函数
-> 数据导入导出
-> 绘图和结果解释
-> 优化算法 / 神经网络专题
```

## 核心概念

### 矩阵与数组运算

`*` 是矩阵乘法，`.*` 是逐元素乘法。很多新手报错都来自把这两个混在一起。

```matlab
A * B   % 矩阵乘法，维度必须匹配
A .* B  % 对应位置相乘，数组大小要兼容
```

### 脚本与函数

脚本依赖当前工作区变量，函数有自己的输入和输出。做可复用算法时，优先把核心逻辑写成函数。

### 路径与数据

运行案例前先确认当前工作目录。很多数据文件找不到，不是文件不存在，而是 Matlab 当前目录不对。

## 典型场景

### 场景一：导入数据

常见入口：

```matlab
load data.mat
T = readtable("data.csv");
X = readmatrix("data.xlsx");
```

读入后先用 `size`、`head`、`summary` 或简单绘图检查数据是否符合预期。

### 场景二：绘图美化

基础图形不要只追求“画出来”，还要补标题、坐标轴、图例：

```matlab
plot(x, y, "LineWidth", 1.5)
grid on
xlabel("x")
ylabel("y")
title("结果曲线")
legend("model")
```

### 场景三：阅读算法案例

建议顺序：

1. 先看 README 和运行入口。
2. 找目标函数或数据读取位置。
3. 找核心参数。
4. 跑一次默认示例。
5. 只改一个参数，再观察结果。

## Matlab 与 Python 对照

| 任务 | Matlab 常见方式 | Python 常见方式 |
| :--- | :--- | :--- |
| 矩阵计算 | 内置矩阵语法 | NumPy |
| 表格数据 | `readtable` | pandas |
| 绘图 | `plot`、`scatter` | matplotlib |
| 机器学习 | 工具箱函数 | scikit-learn / PyTorch |
| 脚本复用 | 函数 `.m` 文件 | 模块和函数 |

## 少量自查

- 你能解释 `*` 和 `.*` 的区别吗？
- 运行案例前，你会检查当前工作目录吗？
- 你知道自己需要哪些工具箱吗？
- 修改算法参数时，你是否一次只改一个关键参数？

## 外部资源

- [MATLAB Onramp](https://matlabacademy.mathworks.com/details/matlab-onramp/gettingstarted)：MathWorks 官方入门课程。
- [MATLAB Documentation](https://www.mathworks.com/help/matlab/)：Matlab 官方文档。
- [Import and Export Data](https://www.mathworks.com/help/matlab/import_export/)：数据导入导出官方文档。
- [MATLAB Plot Gallery](https://www.mathworks.com/products/matlab/plot-gallery.html)：常见绘图示例。

## 下一步

如果你是算法方向，先把 [示例运行指南](RUNNING_EXAMPLES.md) 和 [示例运行索引](EXAMPLE_RUN_INDEX.md) 看完，再进入多目标优化或神经网络目录。先跑通，再理解，再改参数，这条路最稳。
