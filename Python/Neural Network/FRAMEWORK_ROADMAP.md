# 神经网络框架学习入口

本目录大多数示例使用 Python 标准库，是为了讲清原理。理解基本训练循环后，可以继续进入 NumPy、scikit-learn、PyTorch 等工具。

## 为什么要从标准库过渡到框架

标准库示例适合学习：

- 权重、偏置、激活函数。
- 损失和梯度。
- 训练循环。
- 简单分类、回归、序列和卷积直觉。

真实项目通常需要框架，因为框架提供：

- 高效矩阵计算。
- 自动求导。
- 数据集和批训练工具。
- 模型保存加载。
- GPU 支持。

## NumPy：先学矩阵化思维

适合目标：

- 把 Python 循环改成矩阵运算。
- 理解批量输入和批量输出。
- 为后续 PyTorch / TensorFlow 打基础。

建议把 [线性回归梯度下降](examples/linear_regression_gradient_descent.py) 改写成 NumPy 版本。

关注点：

- 输入 `X` 的形状。
- 权重 `W` 的形状。
- 预测 `X @ W + b`。
- 损失如何对整批数据求平均。

## scikit-learn：先跑通机器学习流程

适合目标：

- 分类、回归、数据划分、评估指标。
- 快速建立 baseline。
- 学习训练集和测试集的基本流程。

建议从逻辑回归、SVM、随机森林开始，而不是一上来就深度学习。

典型流程：

```text
准备数据
-> 划分训练集和测试集
-> 选择模型
-> 训练
-> 评估
```

## PyTorch：进入深度学习训练

适合目标：

- 自定义神经网络结构。
- 使用自动求导。
- 学习 CNN、RNN、LSTM、Transformer。
- 保存和加载模型参数。

建议路线：

1. Tensor 和基本运算。
2. `nn.Module`。
3. 损失函数和优化器。
4. DataLoader。
5. 保存和加载模型。

## 已补充的本地过渡示例

| 主题 | 本地入口 | 说明 |
| :--- | :--- | :--- |
| CNN 图像分类直觉 | [tiny_cnn_image_classification.py](examples/tiny_cnn_image_classification.py) | 用 5x5 小图像、卷积特征和简单分类器理解 CNN 思路 |
| 模型保存加载 | [model_save_load_json.py](examples/model_save_load_json.py) | 用 JSON 理解参数保存和恢复 |
| RNN 隐藏状态 | [rnn_hidden_state_demo.py](examples/rnn_hidden_state_demo.py) | 理解序列模型如何携带历史信息 |

## 学习提醒

不要把框架当成魔法。先能解释：

- 输入张量形状是什么。
- 输出表示什么。
- 损失函数在衡量什么。
- 优化器在更新哪些参数。

再去追求更复杂的网络结构。
