# Python 神经网络示例

这个目录保存 Python 版神经网络入门示例。目标不是一上来追求复杂模型，而是先让新手理解神经网络最核心的几个概念：

- 输入特征
- 权重和偏置
- 激活函数
- 损失和误差
- 训练和更新参数
- 分类任务和简单非线性问题

当前示例尽量使用 Python 标准库，方便新手直接运行。后续可以继续扩展 NumPy、scikit-learn、PyTorch 等版本。

如果你不知道该从哪个模型开始，可以先看 [模型选择指南](MODEL_SELECTION.md)。

## 示例列表

| 示例 | 文件 | 说明 | 依赖 |
| :--- | :--- | :--- | :--- |
| 线性回归梯度下降 | [examples/linear_regression_gradient_descent.py](examples/linear_regression_gradient_descent.py) | 用梯度下降学习 `y = 2x + 1` | 标准库 |
| 感知机二分类 | [examples/perceptron_binary_classification.py](examples/perceptron_binary_classification.py) | 用最简单的感知机学习 AND 逻辑 | 标准库 |
| 逻辑回归二分类 | [examples/logistic_regression_binary_classification.py](examples/logistic_regression_binary_classification.py) | 用 Sigmoid 输出概率式分类结果 | 标准库 |
| 简单 MLP 学 XOR | [examples/simple_mlp_xor.py](examples/simple_mlp_xor.py) | 用一个隐藏层的小网络学习 XOR | 标准库 |

## 推荐学习顺序

1. 先运行线性回归示例，理解“预测、误差、梯度、更新参数”。
2. 再运行感知机示例，理解“线性分类”。
3. 再运行逻辑回归示例，理解“概率式二分类”。
4. 再运行 XOR 示例，理解为什么需要隐藏层。
5. 尝试修改学习率、训练轮数、初始权重。
6. 观察输出变化。
7. 再去学习 NumPy、scikit-learn、PyTorch。

## 如何运行

进入仓库根目录后运行：

```powershell
python "Python\Neural Network\examples\linear_regression_gradient_descent.py"
python "Python\Neural Network\examples\perceptron_binary_classification.py"
python "Python\Neural Network\examples\logistic_regression_binary_classification.py"
python "Python\Neural Network\examples\simple_mlp_xor.py"
```

如果你的系统使用 `py` 命令：

```powershell
py "Python\Neural Network\examples\perceptron_binary_classification.py"
```

## 学习重点

### 线性回归与梯度下降

线性回归示例不是完整神经网络，但它展示了训练神经网络时最重要的循环：

```text
预测 -> 计算误差 -> 计算梯度 -> 更新参数
```

理解这一步，再看感知机和 MLP 会轻松很多。

### 感知机

感知机可以理解为一个非常简单的神经元：

```text
输入 -> 加权求和 -> 激活函数 -> 输出
```

它适合解决线性可分的问题，比如 AND、OR。

### 逻辑回归

逻辑回归常用于二分类。它会输出一个 0 到 1 之间的数，可以理解为“属于正类的概率倾向”。

```text
线性加权求和 -> Sigmoid -> 概率式输出
```

### 多层感知机

XOR 不是线性可分问题，单个感知机学不好。加入隐藏层后，网络可以组合多个简单边界，从而学习更复杂的关系。

## 修改建议

新手可以从这些地方开始改：

- `learning_rate`：学习率
- `epochs`：训练轮数
- 训练数据
- 初始权重
- 隐藏层神经元数量

每次只改一个参数，然后观察结果。这样最容易理解参数对训练效果的影响。

## 后续扩展方向

可以继续补充：

- NumPy 手写神经网络
- scikit-learn 分类示例
- PyTorch 入门分类
- CNN 图像分类
- LSTM 时间序列预测
- 模型保存与加载
