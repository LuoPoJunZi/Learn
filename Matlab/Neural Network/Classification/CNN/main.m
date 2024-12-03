%% 初始化
clear         % 清除工作区中的所有变量
close all     % 关闭所有打开的图形窗口
clc           % 清除命令窗口的内容
warning off   % 关闭警告信息

%% 读取数据
res = xlsread('数据集.xlsx'); % 从Excel文件中读取数据，存储在变量res中

%% 分析数据
num_class = length(unique(res(:, end)));  % 计算类别数，假设最后一列为类别标签
num_dim = size(res, 2) - 1;               % 计算特征维度（总列数减去类别列）
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
[P_train, ps_input] = mapminmax(P_train, 0, 1);               % 将训练集输入特征归一化到[0,1]范围
P_test  = mapminmax('apply', P_test, ps_input);               % 使用相同的归一化参数处理测试集输入特征

t_train = categorical(T_train)'; % 将训练集目标输出转换为分类类型，并转置
t_test  = categorical(T_test )'; % 将测试集目标输出转换为分类类型，并转置

%% 数据平铺
% 将数据平铺成1维数据是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
p_train = double(reshape(P_train, num_dim, 1, 1, M)); % 将训练集数据重塑为4D张量，适应CNN输入
p_test  = double(reshape(P_test , num_dim, 1, 1, N)); % 将测试集数据重塑为4D张量，适应CNN输入

%% 构造网络结构
layers = [
    imageInputLayer([num_dim, 1, 1])                           % 输入层，输入尺寸为[num_dim, 1, 1]
    
    convolution2dLayer([2, 1], 16, 'Padding', 'same')          % 卷积层，卷积核大小为2x1，生成16个卷积核，填充方式为'same'
    batchNormalizationLayer                                    % 批归一化层，加速训练和稳定网络
    reluLayer                                                  % ReLU激活层，引入非线性
    
    maxPooling2dLayer([2, 1], 'Stride', [2, 1])               % 最大池化层，池化窗口大小为2x1，步长为[2, 1]
    
    convolution2dLayer([2, 1], 32, 'Padding', 'same')          % 卷积层，卷积核大小为2x1，生成32个卷积核，填充方式为'same'
    batchNormalizationLayer                                    % 批归一化层
    reluLayer                                                  % ReLU激活层
    
    fullyConnectedLayer(num_class)                             % 全连接层，输出神经元数量等于类别数
    softmaxLayer                                               % Softmax层，将输出转换为概率分布
    classificationLayer];                                      % 分类层，计算分类损失
                                        
%% 参数设置
options = trainingOptions('adam', ...                      % 使用Adam优化器
    'MaxEpochs', 500, ...                  % 最大训练次数为500
    'InitialLearnRate', 1e-3, ...          % 初始学习率为0.001
    'L2Regularization', 1e-4, ...          % L2正则化参数，防止过拟合
    'LearnRateSchedule', 'piecewise', ...  % 学习率调度方式为分段
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子为0.1
    'LearnRateDropPeriod', 400, ...        % 每经过400个epoch，学习率下降
    'Shuffle', 'every-epoch', ...          % 每个epoch后打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Plots', 'training-progress', ...      % 绘制训练过程图
    'Verbose', false);                     % 关闭详细训练信息输出

%% 训练模型
net = trainNetwork(p_train, t_train, layers, options); % 使用训练集数据和定义的网络结构训练CNN模型

%% 预测模型
t_sim1 = predict(net, p_train); % 使用训练集数据进行预测，得到预测概率
t_sim2 = predict(net, p_test ); % 使用测试集数据进行预测，得到预测概率

%% 反归一化
% 注意：由于使用的是分类层，predict函数输出的是类别概率，因此无需反归一化
% 这里使用vec2ind可能不适用于categorical类型，可以直接转换为类别索引
T_sim1 = vec2ind(t_sim1'); % 将训练集预测概率转换为类别索引
T_sim2 = vec2ind(t_sim2'); % 将测试集预测概率转换为类别索引

%% 性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ; % 计算训练集准确率
error2 = sum((T_sim2 == T_test )) / N * 100 ; % 计算测试集准确率

%% 绘制网络分析图
analyzeNetwork(layers) % 绘制网络结构图，便于理解网络层级和参数

%% 绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1) % 绘制训练集真实值与预测值对比图
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
title({'训练集预测结果对比'; ['准确率=' num2str(error1) '%']}) % 设置图形标题，显示准确率
grid on % 显示网格

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1) % 绘制测试集真实值与预测值对比图
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
title({'测试集预测结果对比'; ['准确率=' num2str(error2) '%']}) % 设置图形标题，显示准确率
grid on % 显示网格

%% 混淆矩阵
if flag_conusion == 1
    figure
    cm_train = confusionchart(T_train, T_sim1); % 绘制训练集混淆矩阵
    cm_train.Title = '训练集混淆矩阵';
    cm_train.ColumnSummary = 'column-normalized'; % 列归一化显示
    cm_train.RowSummary = 'row-normalized';       % 行归一化显示
        
    figure
    cm_test = confusionchart(T_test, T_sim2); % 绘制测试集混淆矩阵
    cm_test.Title = '测试集混淆矩阵';
    cm_test.ColumnSummary = 'column-normalized'; % 列归一化显示
    cm_test.RowSummary = 'row-normalized';       % 行归一化显示
end
