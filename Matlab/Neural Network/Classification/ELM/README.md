
## ELM分类的用途介绍

### 什么是极限学习机（ELM）？

极限学习机（Extreme Learning Machine，简称ELM）是一种单隐藏层前馈神经网络（Single Hidden Layer Feedforward Neural Network, SLFN）的快速学习算法。与传统的神经网络训练方法不同，ELM在训练过程中随机初始化隐藏层权重和偏置，然后通过最小二乘法（Least Squares Method）直接计算输出权重，极大地加快了训练速度。ELM由于其简单、高效和良好的泛化能力，在各种分类和回归任务中得到了广泛应用。

### ELM的主要用途和应用场景

1. **模式识别**
   - **应用示例**：手写数字识别、人脸识别、指纹识别。
   - **解释**：通过学习输入特征与类别标签之间的关系，ELM能够准确识别和分类不同的模式和图像。

2. **图像分类**
   - **应用示例**：将图像分类为不同类别，如动物、车辆、场景等。
   - **解释**：ELM可以处理高维图像数据，通过特征提取和分类，实现对图像的自动分类。

3. **文本分类**
   - **应用示例**：垃圾邮件检测、情感分析、新闻分类。
   - **解释**：通过将文本数据转换为数值特征（如词频、TF-IDF等），ELM可以学习不同类别之间的区别，实现文本的自动分类。

4. **医疗诊断**
   - **应用示例**：基于病人的症状和检测结果，预测疾病类型，如癌症诊断、糖尿病预测。
   - **解释**：ELM可以辅助医生进行诊断，通过分析大量医疗数据，提高诊断的准确性和效率。

5. **金融风险评估**
   - **应用示例**：信用评分、贷款违约预测、股票价格预测。
   - **解释**：通过分析客户的财务数据和交易行为，ELM可以预测客户的信用风险或潜在的金融风险。

6. **语音识别**
   - **应用示例**：将语音信号转换为文本、语音命令识别。
   - **解释**：ELM能够处理和分类复杂的语音数据，实现准确的语音识别和理解。

7. **预测分析**
   - **应用示例**：销售预测、需求预测、天气预报。
   - **解释**：通过学习历史数据的模式，ELM可以预测未来趋势，辅助企业和组织做出决策。

8. **控制系统**
   - **应用示例**：机器人控制、自动驾驶系统、工业过程控制。
   - **解释**：ELM可以实时处理传感器数据，进行动态控制和调整，提升系统的智能化水平。

### ELM的优势

- **训练速度快**：由于隐藏层权重和偏置是随机初始化且不需要迭代优化，ELM的训练速度远快于传统神经网络。
- **实现简单**：ELM算法实现简单，参数较少，易于编程和调试。
- **良好的泛化能力**：在适当的参数设置下，ELM具有较好的泛化能力，能够有效处理各种分类和回归任务。
- **适应性强**：ELM能够适应不同规模和复杂度的数据集，适用于多种应用场景。

### ELM的局限性

- **随机性依赖**：ELM的性能在一定程度上依赖于隐藏层权重和偏置的随机初始化，可能导致模型结果的不稳定性。
- **过拟合风险**：在处理高维数据或复杂任务时，ELM可能容易过拟合，需要适当的正则化和参数调整。
- **参数选择**：虽然ELM的参数较少，但隐藏层神经元数量的选择对模型性能有显著影响，需要经验或交叉验证进行优化。

### 总结

极限学习机（ELM）作为一种高效的单隐藏层前馈神经网络训练算法，因其快速、简单和良好的泛化能力，在模式识别、图像和文本分类、医疗诊断、金融风险评估、语音识别等多个领域得到了广泛应用。尽管ELM在某些方面存在局限性，但其独特的优势使其在实际问题中仍然是一个重要且有效的工具。通过适当的数据预处理、网络结构设计和参数优化，ELM能够在多种分类和回归任务中表现出色。

---

## 带有详细中文注释的Matlab代码

以下是“ELM分类”神经网络的三个Matlab代码文件，分别为`elmpredict.m`、`elmtrain.m`和`main.m`。每个文件都附有详细的中文注释，帮助您理解每一步的具体操作和实现逻辑。

---

### 1. `elmpredict.m`

```matlab
function Y = elmpredict(p_test, IW, B, LW, TF, TYPE)
% ELPredict - 使用训练好的ELM模型进行预测
%
% 输入参数:
%   p_test - 测试集输入特征矩阵 (R * Q)
%   IW     - 输入权重矩阵 (N * R)
%   B      - 偏置向量 (N * 1)
%   LW     - 输出权重矩阵 (N * S)
%   TF     - 激活函数类型 ('sig' 或 'hardlim')
%   TYPE   - 分类模式 (1) 或 回归模式 (0)
%
% 输出参数:
%   Y      - 预测结果

    %% 计算隐层输出
    Q = size(p_test, 2);                    % 测试样本数量
    BiasMatrix = repmat(B, 1, Q);          % 复制偏置向量以匹配样本数量
    tempH = IW * p_test + BiasMatrix;       % 计算隐层输入

    %% 选择激活函数
    switch TF
        case 'sig'
            H = 1 ./ (1 + exp(-tempH));      % Sigmoid激活函数
        case 'hardlim'
            H = hardlim(tempH);              % 硬限制激活函数
        otherwise
            error('Unsupported transfer function.');
    end

    %% 计算输出
    Y = (H' * LW)';                         % 计算输出层结果

    %% 转化分类模式
    if TYPE == 1
        temp_Y = zeros(size(Y));             % 初始化临时输出矩阵
        for i = 1:size(Y, 2)
            [~, index] = max(Y(:, i));        % 找到概率最大的类别索引
            temp_Y(index, i) = 1;             % 独热编码
        end
        Y = vec2ind(temp_Y);                  % 转换为类别索引
    end
end
```

#### 代码详细说明与注释

#### 1. 函数定义
```matlab
function Y = elmpredict(p_test, IW, B, LW, TF, TYPE)
```
- **功能**：使用训练好的ELM模型对测试数据进行预测。
- **输入参数**：
  - `p_test`：测试集输入特征矩阵，维度为（特征数 * 样本数）。
  - `IW`：输入权重矩阵，维度为（隐藏神经元数 * 特征数）。
  - `B`：偏置向量，维度为（隐藏神经元数 * 1）。
  - `LW`：输出权重矩阵，维度为（隐藏神经元数 * 输出类别数）。
  - `TF`：激活函数类型，可以是'Sigmoid'（'sig'）或'Hard Limit'（'hardlim'）。
  - `TYPE`：分类模式标志，1表示分类，0表示回归。

#### 2. 计算隐层输出
```matlab
Q = size(p_test, 2);                    % 测试样本数量
BiasMatrix = repmat(B, 1, Q);          % 复制偏置向量以匹配样本数量
tempH = IW * p_test + BiasMatrix;       % 计算隐层输入
```
- **功能**：
  - 获取测试样本数量`Q`。
  - 使用`repmat`函数将偏置向量`B`复制`Q`次，形成偏置矩阵`BiasMatrix`。
  - 计算隐层输入`tempH`，即输入权重矩阵`IW`与测试输入`p_test`的乘积，加上偏置矩阵。

#### 3. 选择激活函数
```matlab
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));      % Sigmoid激活函数
    case 'hardlim'
        H = hardlim(tempH);              % 硬限制激活函数
    otherwise
        error('Unsupported transfer function.');
end
```
- **功能**：根据输入参数`TF`选择激活函数类型，对隐层输出`tempH`进行激活，得到激活后输出`H`。
  - `'sig'`：使用Sigmoid激活函数。
  - `'hardlim'`：使用硬限制激活函数。
  - 其他值则报错，提示不支持的激活函数。

#### 4. 计算输出
```matlab
Y = (H' * LW)';                         % 计算输出层结果
```
- **功能**：将激活后的隐层输出`H`与输出权重矩阵`LW`相乘，得到输出层的结果`Y`。

#### 5. 转化分类模式
```matlab
if TYPE == 1
    temp_Y = zeros(size(Y));             % 初始化临时输出矩阵
    for i = 1:size(Y, 2)
        [~, index] = max(Y(:, i));        % 找到概率最大的类别索引
        temp_Y(index, i) = 1;             % 独热编码
    end
    Y = vec2ind(temp_Y);                  % 转换为类别索引
end
```
- **功能**：
  - 如果`TYPE`为1，表示分类任务：
    - 初始化一个与`Y`同尺寸的零矩阵`temp_Y`。
    - 对每个样本，找到输出概率最大的类别索引，并在`temp_Y`中将对应位置置1，形成独热编码。
    - 使用`vec2ind`函数将独热编码转换为类别索引，作为最终的预测结果`Y`。

---

### 2. `elmtrain.m`

```matlab
function [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, N, TF, TYPE)
% ELMTrain - 训练极限学习机模型
%
% 输入参数:
%   p_train - 训练集输入特征矩阵 (R * Q)
%   t_train - 训练集目标输出矩阵 (S * Q)
%   N       - 隐藏层神经元数量
%   TF      - 激活函数类型 ('sig' 或 'hardlim')
%   TYPE    - 回归模式 (0, default) 或 分类模式 (1)
%
% 输出参数:
%   IW  - 输入权重矩阵 (N * R)
%   B   - 偏置向量 (N * 1)
%   LW  - 输出权重矩阵 (N * S)
%   TF  - 激活函数类型
%   TYPE- 模式类型

    % 检查输入数据维度是否匹配
    if size(p_train, 2) ~= size(t_train, 2)
        error('ELM:Arguments', '训练集输入P和输出T的列数必须相同。');
    end

    %% 转入分类模式
    if TYPE == 1
        t_train = ind2vec(t_train);          % 将类别标签转换为独热编码
    end

    %% 初始化权重
    R = size(p_train, 1);                    % 输入特征维度
    Q = size(t_train, 2);                    % 训练样本数量
    IW = rand(N, R) * 2 - 1;                  % 随机初始化输入权重矩阵，范围[-1, 1]
    B  = rand(N, 1);                          % 随机初始化偏置向量
    BiasMatrix = repmat(B, 1, Q);            % 复制偏置向量以匹配样本数量

    %% 计算隐层输出
    tempH = IW * p_train + BiasMatrix;       % 计算隐层输入

    %% 选择激活函数
    switch TF
        case 'sig'
            H = 1 ./ (1 + exp(-tempH));      % Sigmoid激活函数
        case 'hardlim'
            H = hardlim(tempH);              % 硬限制激活函数
        otherwise
            error('Unsupported transfer function.');
    end

    %% 伪逆计算输出权重
    LW = pinv(H') * t_train';                % 计算输出权重矩阵，使用伪逆
end
```

#### 代码详细说明与注释

#### 1. 函数定义
```matlab
function [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, N, TF, TYPE)
```
- **功能**：训练极限学习机（ELM）模型，计算并返回输入权重矩阵`IW`、偏置向量`B`和输出权重矩阵`LW`。
- **输入参数**：
  - `p_train`：训练集输入特征矩阵，维度为（特征数 * 样本数）。
  - `t_train`：训练集目标输出矩阵，维度为（输出类别数 * 样本数）。
  - `N`：隐藏层神经元数量。
  - `TF`：激活函数类型，可以是'Sigmoid'（'sig'）或'Hard Limit'（'hardlim'）。
  - `TYPE`：模式类型，0表示回归，1表示分类。

#### 2. 检查输入数据维度是否匹配
```matlab
if size(p_train, 2) ~= size(t_train, 2)
    error('ELM:Arguments', '训练集输入P和输出T的列数必须相同。');
end
```
- **功能**：确保训练集的输入特征和目标输出在样本数量上匹配。

#### 3. 转入分类模式
```matlab
if TYPE == 1
    t_train = ind2vec(t_train);          % 将类别标签转换为独热编码
end
```
- **功能**：
  - 如果`TYPE`为1，表示分类任务：
    - 使用`ind2vec`函数将类别标签转换为独热编码格式，以适应分类任务的输出要求。

#### 4. 初始化权重
```matlab
R = size(p_train, 1);                    % 输入特征维度
Q = size(t_train, 2);                    % 训练样本数量
IW = rand(N, R) * 2 - 1;                  % 随机初始化输入权重矩阵，范围[-1, 1]
B  = rand(N, 1);                          % 随机初始化偏置向量
BiasMatrix = repmat(B, 1, Q);            % 复制偏置向量以匹配样本数量
```
- **功能**：
  - 获取输入特征维度`R`和训练样本数量`Q`。
  - 随机初始化输入权重矩阵`IW`，范围为[-1, 1]。
  - 随机初始化偏置向量`B`。
  - 使用`repmat`函数将偏置向量复制`Q`次，形成偏置矩阵`BiasMatrix`。

#### 5. 计算隐层输出
```matlab
tempH = IW * p_train + BiasMatrix;       % 计算隐层输入
```
- **功能**：计算隐层的输入`tempH`，即输入权重矩阵`IW`与训练输入`p_train`的乘积，加上偏置矩阵。

#### 6. 选择激活函数
```matlab
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));      % Sigmoid激活函数
    case 'hardlim'
        H = hardlim(tempH);              % 硬限制激活函数
    otherwise
        error('Unsupported transfer function.');
end
```
- **功能**：根据输入参数`TF`选择激活函数类型，对隐层输出`tempH`进行激活，得到激活后输出`H`。
  - `'sig'`：使用Sigmoid激活函数。
  - `'hardlim'`：使用硬限制激活函数。
  - 其他值则报错，提示不支持的激活函数。

#### 7. 伪逆计算输出权重
```matlab
LW = pinv(H') * t_train';                % 计算输出权重矩阵，使用伪逆
```
- **功能**：使用隐层输出矩阵`H`的伪逆（Pseudo-Inverse）与训练目标输出`T_train`相乘，计算输出权重矩阵`LW`。
  - `pinv(H')`：计算`H`的伪逆，用于最小二乘解。
  - `* t_train'`：将伪逆结果与目标输出相乘，得到输出权重矩阵。

---

### 3. `main.m`

```matlab
%% 初始化
clear         % 清除工作区中的所有变量
close all     % 关闭所有打开的图形窗口
clc           % 清除命令窗口的内容
warning off   % 关闭警告信息

%% 读取数据
res = xlsread('数据集.xlsx'); % 从Excel文件中读取数据，存储在变量res中

%% 分析数据
num_class = length(unique(res(:, end)));  % 计算类别数，假设最后一列为类别标签
num_res = size(res, 1);                   % 计算样本总数，每一行代表一个样本
num_size = 0.7;                           % 设置训练集占数据集的比例为70%
res = res(randperm(num_res), :);          % 随机打乱数据集顺序（提高模型泛化能力）
% 如果不需要打乱数据，可以注释掉上行代码
flag_conusion = 1;                        % 标志位为1，启用混淆矩阵绘制（要求Matlab 2018及以上版本）

%% 设置变量存储数据
P_train = [];  % 初始化训练集输入特征
P_test = [];   % 初始化测试集输入特征
T_train = [];  % 初始化训练集目标输出
T_test = [];   % 初始化测试集目标输出

%% 划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 提取当前类别的所有样本
    mid_size = size(mid_res, 1);                    % 当前类别的样本数量
    mid_tiran = round(num_size * mid_size);         % 计算当前类别的训练样本数量（四舍五入）

    % 将当前类别的训练样本添加到训练集
    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入特征（除最后一列）
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集目标输出（最后一列）

    % 将当前类别的测试样本添加到测试集
    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入特征
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集目标输出
end

%% 数据转置
P_train = P_train'; % 转置训练集输入特征，使每列代表一个样本
P_test = P_test';   % 转置测试集输入特征
T_train = T_train'; % 转置训练集目标输出
T_test = T_test';   % 转置测试集目标输出

%% 得到训练集和测试样本个数
M = size(P_train, 2); % 训练集样本数量
N = size(P_test , 2); % 测试集样本数量

%% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);               % 将训练集输入特征归一化到[0,1]范围
p_test  = mapminmax('apply', P_test, ps_input);               % 使用相同的归一化参数处理测试集输入特征
t_train = T_train;                                           % 分类任务中，保持原始标签
t_test  = T_test ;                                           % 分类任务中，保持原始标签

%% 创建模型
num_hiddens = 50;        % 隐藏层节点个数
activate_model = 'sig';  % 激活函数类型
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 1); % 训练ELM模型

%% 仿真测试
T_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE); % 使用训练集进行预测
T_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE); % 使用测试集进行预测

%% 数据排序
[T_train_sorted, index_1] = sort(T_train); % 对训练集真实标签进行排序，并获取排序索引
[T_test_sorted , index_2] = sort(T_test ); % 对测试集真实标签进行排序，并获取排序索引

T_sim1_sorted = T_sim1(index_1); % 根据排序索引重新排列训练集预测结果
T_sim2_sorted = T_sim2(index_2); % 根据排序索引重新排列测试集预测结果

%% 性能评价
error1 = sum((T_sim1_sorted == T_train_sorted)) / M * 100 ; % 计算训练集准确率
error2 = sum((T_sim2_sorted == T_test_sorted )) / N * 100 ; % 计算测试集准确率

%% 绘图
figure
plot(1: M, T_train_sorted, 'r-*', 1: M, T_sim1_sorted, 'b-o', 'LineWidth', 1) % 绘制训练集真实值与预测值对比图
legend('真实值', '预测值') % 添加图例
xlabel('预测样本')       % 设置X轴标签
ylabel('预测结果')       % 设置Y轴标签
title({'训练集预测结果对比'; ['准确率=' num2str(error1) '%']}) % 设置图形标题，显示准确率
grid on % 显示网格

figure
plot(1: N, T_test_sorted, 'r-*', 1: N, T_sim2_sorted, 'b-o', 'LineWidth', 1) % 绘制测试集真实值与预测值对比图
legend('真实值', '预测值') % 添加图例
xlabel('预测样本')       % 设置X轴标签
ylabel('预测结果')       % 设置Y轴标签
title({'测试集预测结果对比'; ['准确率=' num2str(error2) '%']}) % 设置图形标题，显示准确率
grid on % 显示网格

%% 混淆矩阵
figure
cm = confusionchart(T_train, T_sim1); % 绘制训练集混淆矩阵
cm.Title = '训练集混淆矩阵';
cm.ColumnSummary = 'column-normalized'; % 列归一化显示
cm.RowSummary = 'row-normalized';       % 行归一化显示

figure
cm = confusionchart(T_test, T_sim2); % 绘制测试集混淆矩阵
cm.Title = '测试集混淆矩阵';
cm.ColumnSummary = 'column-normalized'; % 列归一化显示
cm.RowSummary = 'row-normalized';       % 行归一化显示
```

#### 代码详细说明与注释

#### 1. 初始化部分
```matlab
clear
close all
clc
warning off
```
- **功能**：清理工作环境，确保代码在干净的环境中运行。
  - `clear`：清除所有变量，释放内存。
  - `close all`：关闭所有打开的图形窗口。
  - `clc`：清空命令窗口内容。
  - `warning off`：关闭所有警告信息，避免干扰输出。

#### 2. 读取数据
```matlab
res = xlsread('数据集.xlsx');
```
- **功能**：从Excel文件中读取数据。
  - 假设数据集位于“数据集.xlsx”文件中，每一行代表一个样本，最后一列为类别标签。

#### 3. 分析数据
```matlab
num_class = length(unique(res(:, end)));
num_res = size(res, 1);
num_size = 0.7;
res = res(randperm(num_res), :);
flag_conusion = 1;
```
- **功能**：
  - 计算类别数`num_class`，即数据集中不同类别的数量。
  - 计算样本总数`num_res`。
  - 设置训练集占数据集的比例`num_size`为70%。
  - 使用`randperm`函数随机打乱数据集顺序，提高模型的泛化能力。
  - 设置标志位`flag_conusion`为1，表示启用混淆矩阵绘制。

#### 4. 设置变量存储数据
```matlab
P_train = []; P_test = [];
T_train = []; T_test = [];
```
- **功能**：初始化训练集和测试集的输入特征`P_train`、`P_test`及目标输出`T_train`、`T_test`。

#### 5. 划分数据集
```matlab
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);
    mid_size = size(mid_res, 1);
    mid_tiran = round(num_size * mid_size);

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];
    T_train = [T_train; mid_res(1: mid_tiran, end)];

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];
end
```
- **功能**：
  - 按类别循环，将每个类别的数据按比例划分为训练集和测试集。
  - 训练集输入特征和目标输出分别存储在`P_train`和`T_train`中。
  - 测试集输入特征和目标输出分别存储在`P_test`和`T_test`中。

#### 6. 数据转置
```matlab
P_train = P_train';
P_test = P_test';
T_train = T_train';
T_test = T_test';
```
- **功能**：将数据转置，使每列代表一个样本，符合Matlab神经网络工具箱的输入要求。

#### 7. 得到训练集和测试样本个数
```matlab
M = size(P_train, 2);
N = size(P_test , 2);
```
- **功能**：获取训练集样本数量`M`和测试集样本数量`N`。

#### 8. 数据归一化
```matlab
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);
t_train = T_train;
t_test  = T_test ;
```
- **功能**：
  - 使用`mapminmax`函数将训练集输入特征`P_train`归一化到[0,1]范围，输出归一化后的训练集`p_train`及归一化参数`ps_input`。
  - 使用相同的归一化参数处理测试集输入特征`P_test`，得到归一化后的测试集`p_test`。
  - 对于分类任务，保持原始标签`T_train`和`T_test`。

#### 9. 创建模型
```matlab
num_hiddens = 50;        % 隐藏层节点个数
activate_model = 'sig';  % 激活函数类型
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 1);
```
- **功能**：
  - 设置隐藏层神经元数量`num_hiddens`为50。
  - 设置激活函数类型`activate_model`为'Sigmoid'（'sig'）。
  - 调用`elmtrain`函数，训练ELM模型，返回输入权重矩阵`IW`、偏置向量`B`、输出权重矩阵`LW`、激活函数类型`TF`和模式类型`TYPE`。

#### 10. 仿真测试
```matlab
T_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE); % 使用训练集进行预测
T_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE); % 使用测试集进行预测
```
- **功能**：
  - 使用训练好的ELM模型对训练集和测试集进行预测，得到预测结果`T_sim1`和`T_sim2`。

#### 11. 数据排序
```matlab
[T_train_sorted, index_1] = sort(T_train);
[T_test_sorted , index_2] = sort(T_test );

T_sim1_sorted = T_sim1(index_1);
T_sim2_sorted = T_sim2(index_2);
```
- **功能**：
  - 对训练集真实标签`T_train`和测试集真实标签`T_test`进行排序，获取排序后的标签`T_train_sorted`和`T_test_sorted`及其排序索引`index_1`和`index_2`。
  - 根据排序索引重新排列预测结果`T_sim1`和`T_sim2`，得到排序后的预测结果`T_sim1_sorted`和`T_sim2_sorted`。

#### 12. 性能评价
```matlab
error1 = sum((T_sim1_sorted == T_train_sorted)) / M * 100 ; % 计算训练集准确率
error2 = sum((T_sim2_sorted == T_test_sorted )) / N * 100 ; % 计算测试集准确率
```
- **功能**：
  - 计算训练集和测试集的分类准确率（百分比）。
  - `error1`：训练集准确率。
  - `error2`：测试集准确率。

#### 13. 绘图
```matlab
figure
plot(1: M, T_train_sorted, 'r-*', 1: M, T_sim1_sorted, 'b-o', 'LineWidth', 1) % 绘制训练集真实值与预测值对比图
legend('真实值', '预测值') % 添加图例
xlabel('预测样本')       % 设置X轴标签
ylabel('预测结果')       % 设置Y轴标签
title({'训练集预测结果对比'; ['准确率=' num2str(error1) '%']}) % 设置图形标题，显示准确率
grid on % 显示网格

figure
plot(1: N, T_test_sorted, 'r-*', 1: N, T_sim2_sorted, 'b-o', 'LineWidth', 1) % 绘制测试集真实值与预测值对比图
legend('真实值', '预测值') % 添加图例
xlabel('预测样本')       % 设置X轴标签
ylabel('预测结果')       % 设置Y轴标签
title({'测试集预测结果对比'; ['准确率=' num2str(error2) '%']}) % 设置图形标题，显示准确率
grid on % 显示网格
```
- **功能**：
  - 绘制训练集和测试集的真实值与预测值对比图，直观展示分类效果。
  - 在图形标题中显示分类准确率。
  - 使用`legend`添加图例，区分真实值和预测值。
  - 设置X轴和Y轴标签，便于理解图形内容。
  - 使用`grid on`显示网格，便于观察数据趋势。

#### 14. 混淆矩阵
```matlab
figure
cm = confusionchart(T_train, T_sim1); % 绘制训练集混淆矩阵
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized'; % 列归一化显示
cm.RowSummary = 'row-normalized';       % 行归一化显示

figure
cm = confusionchart(T_test, T_sim2); % 绘制测试集混淆矩阵
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized'; % 列归一化显示
cm.RowSummary = 'row-normalized';       % 行归一化显示
```
- **功能**：
  - 绘制训练集和测试集的混淆矩阵，评估分类器在不同类别上的表现。
  - 使用`confusionchart`函数创建混淆矩阵图表，显示真实类别与预测类别之间的关系。
  - 设置混淆矩阵的标题`Title`，列和行的归一化显示`ColumnSummary`和`RowSummary`，便于比较分类器在不同类别上的表现。

---

## 总结

通过上述三个Matlab代码文件，您可以实现一个基于极限学习机（ELM）的分类模型。详细的中文注释帮助您理解每一步的具体操作和实现逻辑。ELM模型因其训练速度快、实现简单和良好的泛化能力，在多个领域得到了广泛应用，包括模式识别、图像和文本分类、医疗诊断、金融风险评估以及语音识别等。根据具体需求，您可以调整隐藏层神经元数量、激活函数类型和其他参数，以优化模型性能和适应不同的应用场景。

