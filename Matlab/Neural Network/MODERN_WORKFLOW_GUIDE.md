# Matlab 深度学习现代工作流与旧案例迁移

本目录包含多年积累的 BP、CNN、LSTM、RBF 等案例，其中一些使用 `network`、`train` 或 `trainNetwork`。这些案例仍有学习和复现实验的价值，但 MathWorks 当前推荐的新项目工作流已经转向 `trainnet` 和 `dlnetwork`。

这篇指南不会删除旧案例，而是帮助你判断：旧代码为什么保留，新项目该选什么接口，以及迁移时应该先改哪一层。

## 适合人群

- 能运行本目录案例，但看不懂多种训练接口差异的新手
- 从旧论文或课程代码迁移到较新 Matlab 版本的人
- 准备用 Deep Learning Toolbox 编写新分类、回归或序列模型的人
- 遇到旧函数警告，担心直接替换会改变结果的人

## 学习目标

读完后，你应该能：

- 识别 Matlab 神经网络代码属于哪一代工作流
- 根据任务选择 `fitcnet`、`fitrnet`、`trainnet` 或自定义训练循环
- 理解 `dlnetwork` 在现代工作流中的位置
- 制定保留原始结果、逐步迁移和验证差异的方案
- 避免只替换函数名却忽略数据格式、损失和输出解码

## 三代常见接口

| 代码特征 | 常见对象或函数 | 当前定位 |
| :--- | :--- | :--- |
| 浅层神经网络工作流 | `network`、`feedforwardnet`、`patternnet`、`train` | 旧工作流；MathWorks 已说明 `train` 将在未来版本移除 |
| 早期深度学习工作流 | `trainNetwork`、`SeriesNetwork`、`DAGNetwork`、`layerGraph` | 仍用于理解和运行旧项目，但新开发已不推荐从这里开始 |
| 现代深度学习工作流 | `trainnet`、`dlnetwork`、`minibatchpredict` | 新项目推荐方向，支持更灵活的损失和统一网络对象 |

`trainnet` 从 R2023b 引入。MathWorks 从 R2024a 起将 `trainNetwork` 标记为不推荐，并建议迁移到 `trainnet`。具体可用性仍取决于你的 Matlab 和工具箱版本。

## 为什么仓库仍保留旧代码

保留旧案例不是鼓励新项目继续使用旧接口，而是因为：

- 论文、课程和历史实验常依赖特定 Matlab 版本。
- 直接改写可能改变数据划分、随机初始化、默认预处理和评估结果。
- 旧代码可以帮助理解 BP、训练参数和经典网络结构。
- 有些用户只能使用学校或实验室提供的较旧版本。

因此本仓库采用“双轨策略”：原案例保持可追溯，新学习内容明确指出现代替代方案。需要迁移时，建议复制到新分支或新目录，不要覆盖唯一可复现版本。

## 开始前先检查版本和工具箱

在命令窗口运行：

```matlab
version
ver
which trainnet
which dlnetwork
which trainNetwork
```

判断方式：

- `which trainnet` 能找到文件，说明当前环境提供该接口。
- 命令不存在时，先确认 Matlab 版本和 Deep Learning Toolbox，而不是从网上复制同名函数。
- 同一个仓库的不同案例可能还需要 Statistics and Machine Learning Toolbox、Signal Processing Toolbox 等额外工具箱。

## 按任务选择现代入口

| 任务 | 优先考虑 | 说明 |
| :--- | :--- | :--- |
| 表格特征分类 | `fitcnet` | 适合常规结构化数据，需要 Statistics and Machine Learning Toolbox |
| 表格特征回归 | `fitrnet` | 比手工搭建浅层 `network` 更贴近当前表格建模工作流 |
| 图像、序列和自定义层网络 | `trainnet` | 使用 `dlnetwork`，适合大多数内置训练任务 |
| 自定义损失但仍想用内置训练 | `trainnet` + 自定义损失函数 | 比完整手写训练循环简单 |
| 需要完全控制梯度和更新步骤 | `dlnetwork` + `dlfeval` + `dlgradient` | 适合研究型自定义训练循环 |
| 复现旧论文或课程结果 | 先保留原接口 | 固定环境和随机种子，再单独做迁移对照 |

并不是所有表格问题都应该先上深度学习。样本较少时，还应与线性模型、树模型、SVM 等基线比较。

## `trainNetwork` 到 `trainnet` 的核心变化

旧代码常见形态：

```matlab
options = trainingOptions("adam", ...);
trainedNet = trainNetwork(XTrain, YTrain, layers, options);
YPred = classify(trainedNet, XTest);
```

现代工作流的结构通常是：

```matlab
options = trainingOptions("adam", ...);
net = trainnet(XTrain, YTrain, layers, "crossentropy", options);
scores = minibatchpredict(net, XTest);
YPred = scores2label(scores, classNames);
```

这段代码只展示接口关系，不是可直接套用到所有数据的完整脚本。迁移时必须继续确认：

- `XTrain` 的维度和数据布局是否符合输入层要求。
- `YTrain` 是 categorical、数值数组还是其他格式。
- 损失函数是否与任务和输出层匹配。
- `minibatchpredict` 返回分数的维度。
- `classNames` 的顺序是否和训练标签一致。

现代接口把“训练损失”写得更明确，也把预测分数和标签解码分成两个步骤。不要只替换 `trainNetwork` 为 `trainnet` 后沿用全部旧后处理。

## `dlnetwork` 是什么

`dlnetwork` 是现代深度学习网络对象。它既可以由 `trainnet` 返回，也可以用于自定义训练循环。

你可以把它理解为：

- 保存网络结构和可学习参数。
- 支持训练与预测。
- 与 `dlarray`、自动微分和 GPU 计算配合。
- 为内置训练和自定义训练提供统一基础。

对纯新手而言，先使用 `trainnet`，让 Matlab 管理 mini-batch、优化器和训练进度；只有标准训练流程无法表达目标时，再进入自定义训练循环。

## 迁移旧项目的七个步骤

### 1. 冻结旧环境信息

记录：

```matlab
version
ver
rng default
```

保存旧代码的输入数据、关键参数、输出指标和图形。没有基线结果，就无法判断迁移是否正确。

### 2. 标记旧接口

在项目中搜索：

```text
train(
trainNetwork(
classify(
predict(
SeriesNetwork
DAGNetwork
layerGraph
```

不要看到一个旧函数就立即改。先画出“数据读取 -> 预处理 -> 网络 -> 训练 -> 预测 -> 指标”的完整链路。

### 3. 先保持数据划分不变

训练集、验证集、测试集和归一化方法先不变。一次同时改变 API、数据划分和模型结构，结果不同后很难定位原因。

### 4. 明确损失和指标

分类常见交叉熵，回归常见均方误差，但训练损失不等于最终业务指标。迁移前记录准确率、混淆矩阵、RMSE、MAE 等原始评估方式。

### 5. 替换训练与预测链路

根据官方迁移文档替换网络对象和调用方式。先让最小训练轮数跑通，再恢复完整 epoch 和超参数。

### 6. 比较数值和行为

比较的不只是最终一个数字：

- 输入和输出尺寸
- 训练曲线趋势
- 验证集指标
- 推理速度和内存
- 保存、加载后的预测结果

由于默认初始化和训练实现可能变化，不应期待每次浮点结果逐位相同。更重要的是指标处于合理范围，并且差异可解释。

### 7. 把版本要求写进 README

至少记录：

- 测试过的 Matlab 版本
- 所需工具箱
- 运行入口和数据文件
- 使用旧接口还是现代接口
- 是否固定随机种子
- 迁移前后指标差异

## 从 `train` 浅层网络迁移时的特别注意

`feedforwardnet`、`patternnet` 和 `train` 的数据方向、预处理默认值与现代接口可能不同。旧浅层网络经常使用“特征 x 样本”，而很多现代表格工作流使用“样本 x 特征”。

迁移前先检查：

```matlab
size(XTrain)
size(YTrain)
```

不要为了消除维度报错就不断加转置。先写下每一维代表什么，再按照目标接口文档转换。

对于普通表格分类和回归，优先评估 `fitcnet` 或 `fitrnet`；对于图像、序列或需要深度架构的任务，再使用 `trainnet`。

## 模型保存也要一起迁移

旧项目可能直接保存 `SeriesNetwork` 或 `network` 对象：

```matlab
save("model.mat", "trainedNet")
```

现代项目可以保存 `dlnetwork`，但仍应同时保存推理所需信息：

- 输入归一化参数
- 类别顺序
- 特征名称
- Matlab 和工具箱版本
- 训练配置和随机种子

只有网络权重而没有预处理信息，通常无法稳定复现预测。

## 常见错误

### 只替换函数名

`trainNetwork` 和 `trainnet` 的损失指定、返回对象和预测解码不同。迁移应覆盖训练和推理整条链路。

### 旧代码出现警告就直接删除

警告说明接口方向发生变化，不代表旧实验没有价值。先记录版本和结果，再创建迁移副本。

### 忽略标签顺序

分数矩阵的行或列必须与类别顺序对应。迁移后先打印尺寸和类别，再生成混淆矩阵。

### 把训练损失当成唯一评估

训练损失下降不代表测试集表现良好。分类至少检查独立测试集和混淆矩阵，回归至少检查误差分布与基线。

### 不记录版本

Matlab API 和默认行为会变化。没有版本和工具箱信息，半年后自己也很难复现。

## 完成检查清单

- [ ] 已确认 Matlab 版本和所需工具箱
- [ ] 已识别案例使用 `train`、`trainNetwork` 还是 `trainnet`
- [ ] 旧代码、数据划分和基线结果仍然保留
- [ ] 新代码明确指定损失和输出解码方式
- [ ] 已核对每一维代表样本、特征、时间还是类别
- [ ] 已比较迁移前后的指标和推理结果
- [ ] README 已记录版本、入口、数据和迁移说明

## 官方资料

- [MathWorks：`trainnet`](https://www.mathworks.com/help/deeplearning/ref/trainnet.html)
- [MathWorks：使用内置训练方法训练深度学习神经网络](https://www.mathworks.com/help/deeplearning/ug/builtin-training.html)
- [MathWorks：迁移旧神经网络代码到 `dlnetwork`](https://www.mathworks.com/help/deeplearning/ug/transition-legacy-neural-network-code-to-dlnetwork-workflows.html)
- [MathWorks：`dlnetwork`](https://www.mathworks.com/help/deeplearning/ref/dlnetwork.html)
- [MathWorks：旧 `network.train`](https://www.mathworks.com/help/deeplearning/ref/network.train.html)

## 下一步

先从一个结构简单、运行时间短的分类或回归案例建立基线，再复制出 `trainnet` 版本进行对照。模型选择仍可参考 [神经网络模型选择指南](MODEL_SELECTION.md)，运行和数据排查参考 [Matlab 示例运行指南](../RUNNING_EXAMPLES.md)。
