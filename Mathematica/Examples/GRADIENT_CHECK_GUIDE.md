# 用符号计算检查神经网络梯度

## 适合人群与学习目标

适合已经熟悉列表、替换规则、导数和数值计算的读者。先看 [表达式与函数](../Basics/1.2-expressions-variables-functions.md)、[微积分与数值计算](../Basics/1.6-calculus-numerics.md)，再运行本页案例。

学习目标：把一个微型网络写成表达式，推导损失对参数的梯度，用中心差分核对结果，观察一次梯度下降的效果。

本地入口：[symbolic_gradient_check.wl](symbolic_gradient_check.wl)。只使用本地 Wolfram Language 核心函数，没有数据下载或文件导出。需要可运行的 Mathematica / Wolfram Engine Kernel；这里不依赖 `NetTrain`。

## 核心概念

### 网络、参数与损失

案例使用一个隐藏神经元：

```text
prediction(x) = w2 * tanh(w1*x+b1) + b2
parameters = [w1, b1, w2, b2]
loss = mean((prediction(x_i)-target_i)^2)
```

`tanh` 提供非线性。数据只有 5 个点，目标由一个已知平滑函数产生。损失衡量预测和目标之间的平均平方差。

真实网络参数更多，但仍然需要找到每个参数对损失的变化率。

### 符号微分与自动微分的区别

本例的 `D` 对表达式求导，得到可以继续替换参数值的数学表达式。深度学习框架的自动微分通常沿计算图传播数值导数，适合大规模张量训练。

两者都应用链式法则，但执行方式不同。本例用小表达式展示数学关系和验证方法，不把符号展开当成大型网络的训练工具。

## 推荐阅读顺序

1. `network`：找到输入和四个参数。
2. `trainingData`、`loss`：找到训练数据与均方误差。
3. `gradient`：对参数逐个求导。
4. `initialRules`：把参数列表变成替换规则。
5. `numericalGradient`：计算中心差分。
6. `updatedParameters`、`TestReport`：检查一次更新与梯度结果。

关键表达式如下：

```wolfram
gradient = D[loss, #] & /@ parameters;
initialRules = Thread[parameters -> initialParameters];
symbolicGradient = N[gradient /. initialRules];
```

`/@` 把求导函数应用到每一个参数。`Thread` 把两个同长度列表逐项配成规则，保证参数和值一一对应。

## 为什么还需要中心差分

梯度函数写错、参数顺序混乱或漏掉均值因子，都可能导致训练不正常。可以用独立的数值方法做小规模核对：

```text
gradient_i ~= [loss(theta+epsilon*e_i)-loss(theta-epsilon*e_i)] / (2*epsilon)
```

`e_i` 只在第 `i` 个坐标是 1，因此一次只扰动一个参数。程序使用 `UnitVector` 构造它，用 `epsilon=1e-6` 计算中心差分。

符号梯度与中心差分最大误差应小于本例的 `1e-7` 阈值。这个阈值针对当前数据、参数、尺度和机器精度，不是所有模型的通用标准。

中心差分要计算多次损失，代价随参数量增加，适合调试小案例。它也有截断与舍入误差，不能替代训练时的梯度。

## 如何运行

初学者可以在 Notebook 中直接加载。Windows 路径使用正斜杠，避免反斜杠转义问题：

```wolfram
Get["F:/GithubData/Learn/Mathematica/Examples/symbolic_gradient_check.wl"]
```

已配置好 Kernel 的终端可以从仓库根目录执行：

```powershell
wolframscript -file "Mathematica\Examples\symbolic_gradient_check.wl"
```

脚本打印两种梯度、最大误差、更新前后损失和 `TestReport`，最后返回预测曲线。图形在 Notebook 中最直观。

代码使用 `Module` 把中间符号限制在本次运行内，重复加载时也不会污染笔记本中的同名参数。

## 一次梯度下降做了什么

```text
new_parameters = old_parameters - learning_rate * gradient
```

默认学习率为 0.1，程序检查这一步之后损失下降。图上的两条曲线对应更新前后，黑点是训练数据。

本例只做一步更新，不演示训练收敛，也不能由这一步推出模型在新数据上的效果。

## 验证记录

2026-10-10 的维护环境没有可用 Wolfram Kernel，本 `.wl` 文件尚未实机执行。使用独立的链式法则与中心差分核对相同公式时，最大梯度差约 `4.1e-11`，单步损失由约 `0.253798` 降至 `0.173000`。

这些数值可作为运行时的对照，但独立公式核对不能代替 Wolfram Language 语法、`TestReport` 和图形显示的实机验证。

## 典型场景与常见误区

- 场景：检查手写梯度、推导新损失函数，或在搭建物理约束模型前核对数学公式。
- 不要先给参数永久赋数值再求导。先求符号导数，再用 `/.` 替换。
- 四个参数的顺序必须在符号梯度、数值梯度和更新公式中一致。
- `epsilon` 不是越小越好。太小会让两个相近损失相减时丢失有效数字。
- 学习率过大可能让损失上升。这里的下降检查只针对默认案例，修改参数后失败也可能是在展示这个现象。
- ReLU 等非光滑激活在折点附近不适合直接用相同的中心差分判断；本例选平滑的 `Tanh`。

## 少量修改与理解检查

1. 把 `learningRate` 改成 0.01。参考观察：更新幅度变小，可比较损失下降幅度。
2. 比较 `epsilon=1e-4`、`1e-6`、`1e-10`。参考观察：数值梯度误差不一定单调下降。
3. 调换数值梯度的两个分量。参考观察：梯度一致性检查会发现参数顺序问题。

## 来源与使用提醒

资料核对日期：2026-10-10。源码为按公式原创编写的教学实现。

- [Wolfram D](https://reference.wolfram.com/language/ref/D.html)：符号导数与参数微分。
- [Wolfram VerificationTest](https://reference.wolfram.com/language/ref/VerificationTest.html)：为一个预期结果编写验证。
- [Wolfram TestReport](https://reference.wolfram.com/language/ref/TestReport.html)：汇总多个验证结果。

本地源码遵循仓库 [LICENSE](../../LICENSE)。官方文档和 Mathematica / Wolfram Engine 按其自身许可使用；引文不表示本仓库附带运行环境授权。

## 下一步入口

- [符号化物理约束损失](symbolic_physics_informed_loss.wl)
- [Notebook 学习指南](../NOTEBOOK_GUIDE.md)
- [Python 线性回归与梯度下降](<../../Python/Neural Network/examples/linear_regression_gradient_descent.py>)
