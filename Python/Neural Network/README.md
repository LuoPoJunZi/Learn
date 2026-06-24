# Python 神经网络示例

这个目录保存 Python 版神经网络入门示例。目标不是一上来追求复杂模型，而是先让新手理解神经网络最核心的几个概念：

- 输入特征
- 权重和偏置
- 激活函数
- 损失和误差
- 训练和更新参数
- 分类任务和简单非线性问题

当前示例尽量使用 Python 标准库，方便新手直接运行。理解标准库示例后，可以继续看 [神经网络框架学习入口](FRAMEWORK_ROADMAP.md)，再过渡到 NumPy、scikit-learn、PyTorch 等工具。

如果你不知道该从哪个模型开始，可以先看 [模型选择指南](MODEL_SELECTION.md)。

## 示例列表

| 示例 | 文件 | 说明 | 依赖 |
| :--- | :--- | :--- | :--- |
| 线性回归梯度下降 | [examples/linear_regression_gradient_descent.py](examples/linear_regression_gradient_descent.py) | 用梯度下降学习 `y = 2x + 1` | 标准库 |
| 感知机二分类 | [examples/perceptron_binary_classification.py](examples/perceptron_binary_classification.py) | 用最简单的感知机学习 AND 逻辑 | 标准库 |
| 逻辑回归二分类 | [examples/logistic_regression_binary_classification.py](examples/logistic_regression_binary_classification.py) | 用 Sigmoid 输出概率式分类结果 | 标准库 |
| 简单 MLP 学 XOR | [examples/simple_mlp_xor.py](examples/simple_mlp_xor.py) | 用一个隐藏层的小网络学习 XOR | 标准库 |
| Softmax 多分类 | [examples/softmax_multiclass_classification.py](examples/softmax_multiclass_classification.py) | 用 Softmax 输出多类别概率 | 标准库 |
| RBF 网络回归 | [examples/rbf_network_regression.py](examples/rbf_network_regression.py) | 用径向基函数拟合 `sin(x)` | 标准库 |
| 滑动窗口时间序列预测 | [examples/simple_time_series_prediction.py](examples/simple_time_series_prediction.py) | 用前三个值预测下一个值 | 标准库 |
| 迷你卷积演示 | [examples/mini_cnn_convolution_demo.py](examples/mini_cnn_convolution_demo.py) | 用卷积核提取局部边缘特征 | 标准库 |
| 简单自编码器 | [examples/simple_autoencoder.py](examples/simple_autoencoder.py) | 学习“压缩表示 -> 重建输入”的思路 | 标准库 |
| 过拟合与 L2 正则化 | [examples/overfitting_regularization_demo.py](examples/overfitting_regularization_demo.py) | 对比训练误差和验证误差 | 标准库 |
| 模型保存与加载 | [examples/model_save_load_json.py](examples/model_save_load_json.py) | 用 JSON 保存和恢复模型参数 | 标准库 |
| RNN 隐藏状态演示 | [examples/rnn_hidden_state_demo.py](examples/rnn_hidden_state_demo.py) | 理解序列模型如何携带历史信息 | 标准库 |
| 迷你 CNN 图像分类 | [examples/tiny_cnn_image_classification.py](examples/tiny_cnn_image_classification.py) | 用 5x5 小图像、卷积特征和简单分类器理解 CNN 分类流程 | 标准库 |

## 推荐学习顺序

1. 先运行线性回归示例，理解“预测、误差、梯度、更新参数”。
2. 再运行感知机示例，理解“线性分类”。
3. 再运行逻辑回归示例，理解“概率式二分类”。
4. 再运行 XOR 示例，理解为什么需要隐藏层。
5. 再运行 Softmax 示例，理解多分类概率输出。
6. 再运行 RBF 和时间序列示例，理解不同数据形式下的建模思路。
7. 再运行卷积演示，理解 CNN 为什么能提取局部特征。
8. 再运行自编码器示例，理解压缩表示和重建。
9. 再运行过拟合/正则化示例，理解泛化能力。
10. 再运行模型保存加载示例，理解训练结果如何复用。
11. 再运行 RNN 隐藏状态演示，理解序列模型的记忆机制。
12. 尝试修改学习率、训练轮数、初始权重。
13. 观察输出变化。
14. 再去学习 NumPy、scikit-learn、PyTorch。

## 如何运行

进入仓库根目录后运行：

```powershell
python "Python\Neural Network\examples\linear_regression_gradient_descent.py"
python "Python\Neural Network\examples\perceptron_binary_classification.py"
python "Python\Neural Network\examples\logistic_regression_binary_classification.py"
python "Python\Neural Network\examples\simple_mlp_xor.py"
python "Python\Neural Network\examples\softmax_multiclass_classification.py"
python "Python\Neural Network\examples\rbf_network_regression.py"
python "Python\Neural Network\examples\simple_time_series_prediction.py"
python "Python\Neural Network\examples\mini_cnn_convolution_demo.py"
python "Python\Neural Network\examples\simple_autoencoder.py"
python "Python\Neural Network\examples\overfitting_regularization_demo.py"
python "Python\Neural Network\examples\model_save_load_json.py"
python "Python\Neural Network\examples\rnn_hidden_state_demo.py"
python "Python\Neural Network\examples\tiny_cnn_image_classification.py"
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

### Softmax 多分类

二分类通常只需要判断“是或否”，多分类则要在多个类别中选择一个。Softmax 会把多个得分转换成一组概率式输出。

```text
多个类别得分 -> Softmax -> 每个类别的概率
```

### RBF 网络

RBF 网络用“离某些中心点有多近”来生成特征，适合帮助新手理解另一类神经网络结构。

```text
输入 -> 到中心点的距离 -> RBF 特征 -> 输出
```

### 时间序列预测

时间序列示例使用滑动窗口，把连续数据转换成监督学习样本：

```text
前三个值 -> 下一个值
```

### 迷你卷积演示

卷积演示不训练模型，只展示 CNN 中最关键的一步：用小卷积核扫描局部区域，提取边缘或纹理特征。

```text
小矩阵图像 -> 卷积核扫描 -> 特征响应
```

### 自编码器

自编码器学习把输入压缩成更小的隐藏表示，再从隐藏表示重建原始输入。

```text
输入 -> 压缩表示 -> 重建输出
```

### 过拟合与正则化

过拟合指模型在训练数据上表现很好，但在没见过的数据上表现变差。正则化会限制模型参数，帮助模型更关注可泛化的规律。

### 模型保存与加载

训练得到的权重和偏置可以保存下来，之后再加载使用。真实框架有专门格式，这里的 JSON 示例用于理解基本思想。

### RNN 隐藏状态

RNN 会一步一步处理序列，并用隐藏状态携带前面时间步的信息。这个示例只演示前向传播，不做训练。

## 参考来源

本目录案例参考了以下官方教程的主题方向，并改写为纯 Python 标准库教学示例：

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Save and Load the Model](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
- [TensorFlow Basic classification](https://www.tensorflow.org/tutorials/keras/classification)
- [TensorFlow Overfit and underfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
- [TensorFlow Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [TensorFlow Intro to Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder)

## 修改建议

新手可以从这些地方开始改：

- `learning_rate`：学习率
- `epochs`：训练轮数
- 训练数据
- 初始权重
- 隐藏层神经元数量

每次只改一个参数，然后观察结果。这样最容易理解参数对训练效果的影响。

## 扩展学习入口

原来计划扩展的方向已经补成入口或本地示例：

- NumPy、scikit-learn、PyTorch 学习路线：[FRAMEWORK_ROADMAP.md](FRAMEWORK_ROADMAP.md)
- CNN 图像分类直觉：[tiny_cnn_image_classification.py](examples/tiny_cnn_image_classification.py)
- RNN / LSTM 前置直觉：[rnn_hidden_state_demo.py](examples/rnn_hidden_state_demo.py)
- 模型保存与加载：[model_save_load_json.py](examples/model_save_load_json.py)
