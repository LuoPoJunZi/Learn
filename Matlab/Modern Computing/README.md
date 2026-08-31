# Matlab 现代计算案例

这个目录补充三个与现代 AI 和科学机器学习相关的轻量案例。代码优先使用基础矩阵运算，让没有额外工具箱的新手也能先理解核心公式。

## 适合人群

- 已学习 Matlab 向量、矩阵、脚本和基础绘图的读者。
- 想理解 Transformer 自注意力计算的人。
- 听说过 Physics-Informed Neural Network（PINN），但不理解物理残差如何进入损失函数的人。
- 想理解 Fourier Neural Operator（FNO）为什么在频域学习函数到函数映射的人。

## 案例索引

| 案例 | 入口 | 学习重点 | 依赖 |
| :--- | :--- | :--- | :--- |
| 因果自注意力 | [self_attention_demo.m](self_attention_demo.m) | Query、Key、Value、缩放、Softmax、因果遮罩 | 基础 Matlab |
| 物理约束损失 | [physics_informed_loss_demo.m](physics_informed_loss_demo.m) | 数据损失、ODE 残差、损失权重、参数搜索 | 基础 Matlab |
| Fourier Neural Operator 直觉 | [fourier_neural_operator_demo.m](fourier_neural_operator_demo.m) | FFT、频域乘子、模态截断、函数空间映射 | 基础 Matlab |

## 如何运行

1. 在 Matlab 中把当前文件夹切换到 `Matlab/Modern Computing`。
2. 在命令窗口执行：

```matlab
self_attention_demo
physics_informed_loss_demo
fourier_neural_operator_demo
```

三个脚本都会打印关键数值并生成图形，不会修改外部文件。

## 案例一：因果自注意力

脚本按下面顺序完成一次前向计算：

```text
Token 向量
  -> 线性映射得到 Query / Key / Value
  -> Query 与 Key 计算相似度
  -> 除以 sqrt(d) 控制数值范围
  -> 遮住未来位置
  -> Softmax 得到注意力权重
  -> 权重乘 Value 得到上下文表示
```

这里没有训练权重，也没有多头注意力、位置编码、残差连接和归一化，所以它不是完整 Transformer。它只负责展示最核心的一次注意力计算。

## 案例二：物理约束损失

脚本研究简单微分方程：

```text
du/dt + u = 0,  u(0) = 1
```

候选函数写成 `u(t) = exp(-rate * t)`。程序同时计算：

- 数据损失：预测值和少量测量值之间的均方误差。
- 物理损失：`du/dt + u` 离零有多远。
- 组合损失：`dataLoss + physicsWeight * physicsLoss`。

真实 PINN 会用神经网络表示 `u(t)`，并通过自动微分和优化器更新网络参数。本案例只用一个参数和网格搜索，帮助你先理解损失函数结构。

## 案例三：Fourier Neural Operator 直觉

脚本为一维扩散方程合成多组“初始函数 -> 下一时刻函数”训练数据，然后在 Fourier 空间中用最小二乘学习每个频率的乘子。它展示了三个关键直觉：

- 神经算子学习的是函数之间的映射，而不只是固定长度向量之间的映射。
- Fourier 变换把全局空间模式分解成频率模态。
- 只保留有限低频模态可以降低计算量，但会丢失高频细节。

真正的 FNO 会在多个通道之间学习谱卷积权重，并叠加逐点线性层、非线性激活和多个 Fourier Block。本案例只学习一个可解释的对角频域算子，不是完整神经网络。

## 推荐修改点

- 在自注意力案例中移除 `causalMask`，比较双向注意力和因果注意力。
- 修改 Token 向量或随机种子，观察注意力权重变化。
- 修改 `physicsWeight`，观察数据拟合与物理约束之间的平衡。
- 增大测量噪声，比较仅看数据和加入物理约束后的估计结果。
- 修改 FNO 案例的 `retainedModes`，观察速度思想与高频误差之间的取舍。
- 修改 `diffusivity` 和 `deltaTime`，观察扩散过程如何衰减不同频率。

## 权威参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [MathWorks attention 文档](https://www.mathworks.com/help/deeplearning/ref/dlarray.attention.html)
- [MathWorks Physics-Informed Machine Learning](https://www.mathworks.com/help/deeplearning/physics-informed-machine-learning.html)
- [Physics-informed neural networks, Journal of Computational Physics](https://doi.org/10.1016/j.jcp.2018.10.045)
- [MathWorks: Solve PDE Using Fourier Neural Operator](https://www.mathworks.com/help/deeplearning/ug/solve-pde-using-fourier-neural-operator.html)
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

本目录脚本为依据公式重新编写的原创教学示例，没有复制 MathWorks 示例源码。

## 下一步

- [Matlab 基础教程](../Basics/README.md)
- [Matlab 神经网络案例](<../Neural Network/README.md>)
- [Matlab 案例阅读指南](../CASE_READING_GUIDE.md)
