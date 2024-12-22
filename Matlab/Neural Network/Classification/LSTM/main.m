%% 初始化
clear              % 清除工作区中的所有变量，确保没有残留变量影响结果
close all          % 关闭所有打开的图形窗口，确保绘图环境的干净
clc                % 清空命令行窗口，提升可读性
warning off        % 关闭所有警告信息，避免运行过程中显示不必要的警告

%% 读取数据
res = xlsread('数据集.xlsx');  % 从Excel文件中读取数据，假设数据的最后一列为类别标签
% res变量存储了读取的数据，数据应按照时间顺序排列

%% 分析数据
num_class = length(unique(res(:, end)));  % 获取类别数，假设数据的最后一列是类别标签
num_dim = size(res, 2) - 1;               % 特征维度，即总列数减去类别列
num_res = size(res, 1);                   % 样本数，即数据的行数
num_size = 0.7;                           % 训练集占数据集的比例，这里设定为70%
res = res(randperm(num_res), :);          % 打乱数据集顺序（如果不需要打乱数据集，注释该行）
flag_conusion = 1;                        % 是否绘制混淆矩阵的标志（1为绘制，0为不绘制）

%% 设置变量存储数据
P_train = []; P_test = [];  % 输入数据：训练集和测试集
T_train = []; T_test = [];  % 输出数据：训练集和测试集

%% 划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 提取当前类别的数据
    mid_size = size(mid_res, 1);                    % 当前类别的样本个数
    mid_train_size = round(num_size * mid_size);    % 该类别的训练样本数
    
    % 划分训练集和测试集
    P_train = [P_train; mid_res(1: mid_train_size, 1: end - 1)];  % 训练集输入数据
    T_train = [T_train; mid_res(1: mid_train_size, end)];         % 训练集输出数据
    
    P_test  = [P_test; mid_res(mid_train_size + 1: end, 1: end - 1)];  % 测试集输入数据
    T_test  = [T_test; mid_res(mid_train_size + 1: end, end)];         % 测试集输出数据
end
% 通过逐类别划分，确保训练集和测试集在每个类别中都有代表性

%% 数据转置
P_train = P_train'; P_test = P_test';  % 转置输入数据，使每列为一个样本
T_train = T_train'; T_test = T_test';  % 转置输出数据，使每列为一个样本

%% 得到训练集和测试样本个数
M = size(P_train, 2);  % 训练集样本数
N = size(P_test , 2);  % 测试集样本数

%% 数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);  % 对训练集输入特征进行归一化，范围[0,1]
P_test  = mapminmax('apply', P_test, ps_input);  % 使用训练集的归一化参数对测试集输入特征进行归一化

t_train = categorical(T_train)';  % 将训练集输出数据转换为分类类型，并转置
t_test  = categorical(T_test )';  % 将测试集输出数据转换为分类类型，并转置

%% 数据平铺
% 将数据平铺为1维数据（可选择平铺为2维或3维数据，需调整模型结构）
P_train = double(reshape(P_train, num_dim, 1, 1, M));  % 训练集数据平铺为4维数组 (特征维度 × 1 × 1 × 样本数)
P_test  = double(reshape(P_test , num_dim, 1, 1, N));  % 测试集数据平铺为4维数组 (特征维度 × 1 × 1 × 样本数)

%% 数据格式转换
% 将数据按样本重新整理为单元格数组，适应trainNetwork函数的输入要求
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);  % 训练集数据存储，每个单元格包含一个样本的特征向量
end

for i = 1 : N
    p_test{i, 1}  = P_test(:, :, 1, i);  % 测试集数据存储，每个单元格包含一个样本的特征向量
end

%% 建立 LSTM 网络模型
layers = [
    sequenceInputLayer(num_dim)  % 输入层，输入数据的维度是特征数
    
    lstmLayer(6, 'OutputMode', 'last')  % LSTM层，包含6个LSTM单元，输出最后一个时间步的结果
    reluLayer                         % ReLU激活层，增加网络的非线性
    
    fullyConnectedLayer(num_class)     % 全连接层，输出为类别数
    softmaxLayer                      % Softmax层，将输出转换为概率分布
    classificationLayer];              % 分类层，计算损失并进行分类

%% 参数设置
options = trainingOptions('adam', ...      % 选择Adam优化算法
    'MaxEpochs', 1000, ...                 % 最大训练迭代次数设为1000
    'InitialLearnRate', 0.01, ...          % 初始学习率设为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降策略设为分段下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子设为0.1
    'LearnRateDropPeriod', 750, ...        % 每750次迭代后学习率下降
    'Shuffle', 'every-epoch', ...          % 每个epoch结束后打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证，即不进行验证
    'L2Regularization', 1e-4, ...          % L2正则化参数设为1e-4，防止过拟合
    'Plots', 'training-progress', ...      % 绘制训练过程的进度图
    'Verbose', false);                     % 关闭训练过程中的详细信息显示

%% 训练模型
net = trainNetwork(p_train, t_train, layers, options);  % 使用训练数据训练LSTM网络

%% 预测模型
t_sim1 = predict(net, p_train);  % 对训练集数据进行预测，得到训练集预测结果
t_sim2 = predict(net, p_test );  % 对测试集数据进行预测，得到测试集预测结果

%% 反归一化
T_sim1 = vec2ind(t_sim1');  % 将训练集预测结果的概率向量转换为类别索引
T_sim2 = vec2ind(t_sim2');  % 将测试集预测结果的概率向量转换为类别索引

%% 性能评价
error1 = sum((T_sim1 == T_train)) / M * 100;  % 计算训练集的分类准确率（百分比）
error2 = sum((T_sim2 == T_test )) / N * 100;  % 计算测试集的分类准确率（百分比）

%% 绘制网络分析图
analyzeNetwork(layers);  % 可视化和分析LSTM网络的结构

%% 绘图
% 绘制训练集和测试集预测结果对比图

% 绘制训练集预测结果对比图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1) % 绘制训练集真实类别与预测类别的对比曲线，红色星形标记为真实值，蓝色圆形标记为预测值
legend('真实值', '预测值')                                        % 添加图例，区分真实值和预测值
xlabel('预测样本')                                                % 设置X轴标签为“预测样本”
ylabel('预测结果')                                                % 设置Y轴标签为“预测结果”
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};  % 创建标题字符串，包括准确率
title(string)                                                    % 添加图形标题
grid                                                             % 显示网格，提升图形的可读性

% 绘制测试集预测结果对比图
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)   % 绘制测试集真实类别与预测类别的对比曲线，红色星形标记为真实值，蓝色圆形标记为预测值
legend('真实值', '预测值')                                        % 添加图例，区分真实值和预测值
xlabel('预测样本')                                                % 设置X轴标签为“预测样本”
ylabel('预测结果')                                                % 设置Y轴标签为“预测结果”
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};   % 创建标题字符串，包括准确率
title(string)                                                    % 添加图形标题
grid                                                             % 显示网格，提升图形的可读性

%% 混淆矩阵
if flag_conusion == 1
    % 绘制训练集的混淆矩阵
    figure
    cm = confusionchart(T_train, T_sim1);                % 创建训练集的混淆矩阵
    cm.Title = '训练集混淆矩阵';                             % 设置混淆矩阵标题
    cm.ColumnSummary = 'column-normalized';               % 设置列摘要为列归一化
    cm.RowSummary = 'row-normalized';                     % 设置行摘要为行归一化
    
    % 绘制测试集的混淆矩阵
    figure
    cm = confusionchart(T_test, T_sim2);                 % 创建测试集的混淆矩阵
    cm.Title = '测试集混淆矩阵';                             % 设置混淆矩阵标题
    cm.ColumnSummary = 'column-normalized';               % 设置列摘要为列归一化
    cm.RowSummary = 'row-normalized';                     % 设置行摘要为行归一化
end
