# Flow Matching：学习速度，再生成样本

## 适合人群与学习目标

适合会使用矩阵、反斜杠最小二乘、函数和基础绘图的读者。先读 [Matlab 基础教程](../Basics/README.md)，再看 [现代计算案例索引](README.md)。

目标是看清生成建模中的“路径、速度监督、回归、ODE 采样”四个步骤，并区分训练样本的配对路径和生成时的整体速度场。

代码入口：[flow_matching_demo.m](flow_matching_demo.m)。只依赖基础 Matlab，使用 `ode45`，不需要 Deep Learning Toolbox，不访问网络或导出文件。脚本使用局部函数与隐式扩展，建议 Matlab R2016b 或更新版本。

## 一、先理解要生成什么

本例源分布是二维标准正态噪声。目标分布是均值 `[2,-1]`、各维标准差 `[0.6,1.3]` 的二维正态分布。目标很简单，便于同时进行数值验证。

生成模型学习的是把一批新噪声转成与目标数据分布相似的样本。它无需把某个新噪声对应到一个预先指定的目标点。

## 二、从端点构造速度监督

训练阶段独立抽取 `x0` 和 `x1`，前者来自源分布，后者来自目标分布。选择时间 `t`，构造直线路径：

```text
x_t = (1-t)*x0 + t*x1
u_t = d(x_t)/dt = x1-x0
```

`x_t` 是当前的位置，`u_t` 是这对训练端点对应的速度。学习任务是让模型根据 `(t,x_t)` 输出速度，尽量接近 `u_t`。

本例在 61 个时间节点上分别拟合线性模型：

```matlab
features = [positions, ones(trainingCount, 1)];
coefficients = features \ velocityTargets;
```

`features` 是 `4000 x 3`，两列坐标加一列偏置；`velocityTargets` 是 `4000 x 2`；系数是 `3 x 2`，分别预测两个方向的速度。时间节点之间用 `interp1` 插值系数。

真正的神经 Flow Matching 常用一个带时间输入的网络，并抽取随机时间训练。本例使用时间网格和最小二乘回归，让速度学习本身可以直接观察。

### 为什么训练误差不会降到零

独立配对时，同一位置附近可能来自不同端点组合，对应不同速度。只看 `(t,x)` 的模型不能恢复每个隐藏的端点对。

平方误差回归学习的是条件平均速度 `E[x1-x0 | x_t=x]`。本例的非零训练误差包含这个不可消除的差异，不能简单解释为没有学会生成。

## 三、从新噪声开始解 ODE

完成训练后，另生成 300 个新噪声点。生成阶段只有这些新点与已学系数：

```text
dx/dt = learned_velocity(t,x)
start: t=0, x=new_noise
finish: t=1, x=generated_sample
```

`ode45` 反复调用已学速度函数完成积分。训练用的目标点没有进入采样函数，所以这里不是把目标坐标直接传给生成器。

训练端点间的条件路径是直线，但生成阶段沿边际速度场运动，轨迹可以弯曲；二者不能混为一谈。`ode45` 的输出时间点数量也不等于其内部求解步数。

## 四、运行与检查

在 Matlab 中切换到本目录，再执行：

```matlab
diagnostics = flow_matching_demo();
```

从仓库根目录的终端也可以运行。命令先添加案例目录，再调用函数：

```powershell
matlab -batch "addpath('Matlab/Modern Computing'); flow_matching_demo();"
```

结果包括目标和生成样本的均值、标准差、训练 MSE、解析 ODE 端点误差以及已学端点 RMSE。图左侧比较噪声、目标数据与生成样本；右侧显示 12 条已学轨迹。

有限样本均值和标准差不会完全等于总体参数。增加样本数量可以帮助观察这个差异，但不要把一次近似吻合当成完整的生成质量评估。

默认参数在 Matlab R2023a 的一次实测：

| 指标 | 结果 |
| :--- | :--- |
| 生成样本均值 | `[1.920,-0.942]` |
| 生成样本标准差 | `[0.588,1.329]` |
| 解析 ODE 端点最大误差 | 约 `2.245e-8` |
| 已学端点 RMSE | 约 `0.032007` |

RMSE 对照的是同一新噪声在解析流下的终点，不是和任意训练目标点比较。不同运行环境可能出现细小浮点差异。

### 解析速度场为什么有用

独立 Gaussian 配对在时间 `t` 的方差和平均速度有解析形式，各维分别计算：

```text
variance(t) = (1-t)^2 + t^2*sigma^2
slope(t) = (t*sigma^2 - (1-t)) / variance(t)
v(t,x) = mu + slope(t)*(x-t*mu)
```

同一初始点的精确轨迹是 `t*mu + sqrt(variance(t))*x0`，终点是 `mu+sigma*x0`。

程序单独积分解析速度场并检查终点误差，目的是核对 ODE 的形状与数值精度。已学采样始终使用回归系数；解析结果只做对照。

推荐阅读顺序：训练端点和路径 -> `features \ velocityTargets` -> `learned_velocity()` -> `ode45` -> `gaussian_velocity()` 的解析对照。

## 典型场景与常见误区

- 场景：在学习图像、音频生成模型之前，建立连续时间生成建模的数学直觉。
- 模型只针对简单 Gaussian 分布使用线性速度回归，不表示线性模型可以生成任意复杂图像。
- 本例独立配对端点，没有计算最优传输配对；不要把直线插值直接称为已求解的最优传输。
- 不要把每个训练端点对的速度直接用于新点采样，需要先学习只依赖时间和位置的速度函数。
- 修改采样数量时，状态重排的行数也必须同步。源码通过 `sampleCount` 传递，避免矩阵维度混淆。

## 少量修改与理解检查

1. 改 `targetMean`。参考现象：生成点云中心随之移动。
2. 改 `targetStd`。参考现象：两个坐标方向的点云宽度改变。
3. 减少 `trainingCount`。参考现象：速度回归与解析场的差异通常更大，但不保证每次单调变化。

## 来源与使用提醒

资料核对日期：2026-10-10。源码按公式原创编写，没有引入外部库或模型权重。

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)：理解概率路径和速度场回归。
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)：继续学习连续与离散方法的系统导读。
- [Meta 官方 Flow Matching 文档](https://facebookresearch.github.io/flow_matching/)：进一步了解 PyTorch 实现。对应 [仓库 LICENSE](https://github.com/facebookresearch/flow_matching/blob/main/LICENSE) 是 CC BY-NC 4.0，含非商业限制，复用前须核对许可。
- [MathWorks ode45](https://www.mathworks.com/help/matlab/ref/ode45.html)：查询求解器输入、容差与返回值。

本地案例使用仓库 [LICENSE](../../LICENSE)。外部参考资料与仓库有各自许可，不能用本仓库许可覆盖它们。

## 下一步入口

- [物理约束损失](physics_informed_loss_demo.m)
- [Fourier Neural Operator 直觉](fourier_neural_operator_demo.m)
- [Matlab 深度学习现代工作流](<../Neural Network/MODERN_WORKFLOW_GUIDE.md>)
