# 神经网络模型选择指南

这个指南帮助你根据任务类型选择 `Neural Network` 目录下的模型案例。

## 先判断任务类型

| 你的目标 | 任务类型 | 推荐目录 |
| :--- | :--- | :--- |
| 判断属于哪一类 | 分类 | [Classification](Classification/) |
| 预测连续数值 | 回归 | [Regression](Regression/) |
| 根据历史数据预测未来 | 时间序列预测 | [Time Series Prediction](<Time Series Prediction/>) |

## 按数据类型选择模型

| 数据类型 | 推荐模型 | 说明 |
| :--- | :--- | :--- |
| 表格数据 | BP、SVM、RF、ELM | 适合结构化特征 |
| 图片数据 | CNN | 适合图像分类或图像特征学习 |
| 时间序列 | LSTM、BP | LSTM 更适合有时间依赖的数据 |
| 小样本数据 | SVM、RF、RBF | 通常比深层网络更稳 |
| 非线性关系明显 | BP、RBF、SVM | 能拟合复杂非线性关系 |
| 想加入优化算法调参 | GA-BP、PSO-BP | 用遗传算法或粒子群优化 BP 参数 |

## 模型速查

| 模型 | 中文说明 | 适合任务 | 特点 |
| :--- | :--- | :--- | :--- |
| BP | 反向传播神经网络 | 分类、回归、预测 | 经典基础模型，适合入门 |
| CNN | 卷积神经网络 | 图像分类、特征提取 | 擅长处理图像和局部特征 |
| ELM | 极限学习机 | 分类、回归 | 训练速度快，结构相对简单 |
| GA-BP | 遗传算法优化 BP | 分类、回归、预测 | 用 GA 优化 BP 初始权重或参数 |
| LSTM | 长短期记忆网络 | 时间序列预测 | 擅长处理时间依赖 |
| PSO-BP | 粒子群优化 BP | 分类、回归、预测 | 用 PSO 优化 BP 参数 |
| RBF | 径向基函数网络 | 分类、回归 | 适合非线性拟合 |
| RF | 随机森林 | 分类、回归 | 稳定、抗过拟合能力较好 |
| SVM | 支持向量机 | 分类、回归 | 小样本和高维特征常用 |

## 分类任务怎么选

推荐路径：

1. 先看 [Classification/BP](Classification/BP/)：理解数据划分、归一化、训练和预测流程。
2. 表格数据可以看 SVM、RF、ELM。
3. 图像数据优先看 CNN。
4. 想看优化算法调参，可以看 GA-BP、PSO-BP。

## 回归任务怎么选

推荐路径：

1. 先看 [Regression/BP](Regression/BP/)。
2. 如果数据是表格特征，可以比较 BP、SVM、RF。
3. 如果非线性强，可以看 RBF 或 SVM。
4. 如果想优化 BP 参数，可以看 GA-BP、PSO-BP。

## 时间序列预测怎么选

推荐路径：

1. 先看 [Time Series Prediction/LSTM](<Time Series Prediction/LSTM/>)。
2. 如果只是简单趋势预测，可以比较 BP。
3. 如果数据量不大，可以尝试 SVM、RF、RBF。
4. 注意时间序列不要随机打乱训练和测试顺序，避免数据泄漏。

## 运行案例前检查

1. 当前工作目录是否在模型案例目录下。
2. 数据文件是否存在，例如 `.xlsx` 或 `.mat`。
3. 是否安装所需工具箱。
4. 输入特征和标签列是否和代码假设一致。
5. 分类任务的标签是否从 1 开始，是否需要转换。
6. 是否需要设置随机种子，方便复现实验。

## 常见工具箱需求

| 模型 | 可能需要的工具箱 |
| :--- | :--- |
| BP、CNN、LSTM | Deep Learning Toolbox |
| SVM、RF | Statistics and Machine Learning Toolbox |
| GA-BP | Global Optimization Toolbox 或相关自定义代码 |
| PSO-BP | 自定义 PSO 代码或优化工具箱 |

具体以每个案例代码为准。
