%% 初始化
clear              % 清除工作区中的变量
close all          % 关闭所有图形窗口
clc                % 清除命令窗口
warning off        % 关闭警告信息

%%  读取数据
res = xlsread('数据集.xlsx');  % 从 Excel 文件中读取数据

%%  分析数据
num_class = length(unique(res(:, end)));  % 获取类别数（Excel 最后一列是类别）
num_dim = size(res, 2) - 1;               % 特征维度（去掉类别列）
num_res = size(res, 1);                   % 样本数（行数）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（如果不需要打乱数据集，注释该行）
flag_conusion = 1;                        % 是否绘制混淆矩阵的标志（2018版本及以上需要）

%%  设置变量存储数据
P_train = []; P_test = [];  % 输入数据：训练集和测试集
T_train = []; T_test = [];  % 输出数据：训练集和测试集

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 提取当前类别的数据
    mid_size = size(mid_res, 1);                    % 当前类别的样本个数
    mid_tiran = round(num_size * mid_size);         % 该类别的训练样本数

    % 划分训练集和测试集
    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];  % 训练集输入数据
    T_train = [T_train; mid_res(1: mid_tiran, end)];         % 训练集输出数据

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入数据
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出数据
end

%%  数据转置
P_train = P_train'; P_test = P_test';  % 转置输入数据
T_train = T_train'; T_test = T_test';  % 转置输出数据

%%  得到训练集和测试样本个数
M = size(P_train, 2);  % 训练集样本数
N = size(P_test , 2);  % 测试集样本数

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);  % 对训练集数据进行归一化
P_test  = mapminmax('apply', P_test, ps_input);  % 应用训练集的归一化参数到测试集

t_train =  categorical(T_train)';  % 转换为分类类型
t_test  =  categorical(T_test )';  % 转换为分类类型

%%  数据平铺
% 将数据平铺为1维数据（可选择平铺为2维或3维数据，需调整模型结构）
P_train = double(reshape(P_train, num_dim, 1, 1, M));  % 训练集数据平铺
P_test  = double(reshape(P_test , num_dim, 1, 1, N));  % 测试集数据平铺

%%  数据格式转换
% 将数据按样本重新整理
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);  % 训练集数据存储
end

for i = 1 : N
    p_test{i, 1}  = P_test(:, :, 1, i);  % 测试集数据存储
end

%%  建立 LSTM 网络模型
layers = [
    sequenceInputLayer(num_dim)  % 输入层，输入数据的维度是特征数

    lstmLayer(6, 'OutputMode', 'last')  % LSTM 层，6 个单元，输出最后一个时间步的结果
    reluLayer                         % ReLU 激活层

    fullyConnectedLayer(num_class)     % 全连接层，输出为类别数
    softmaxLayer                      % Softmax 层，用于分类
    classificationLayer];              % 分类层，用于输出预测类别

%%  参数设置
options = trainingOptions('adam', ...      % Adam 优化算法
    'MaxEpochs', 1000, ...                 % 最大迭代次数
    'InitialLearnRate', 0.01, ...          % 初始学习率
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降策略
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子
    'LearnRateDropPeriod', 750, ...        % 每 750 次训练后学习率下降
    'Shuffle', 'every-epoch', ...          % 每个 epoch 打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'L2Regularization', 1e-4, ...          % L2 正则化参数
    'Plots', 'training-progress', ...      % 绘制训练过程曲线
    'Verbose', false);                     % 关闭训练过程信息显示

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);  % 使用训练数据训练 LSTM 网络

%%  预测模型
t_sim1 = predict(net, p_train);  % 对训练集进行预测
t_sim2 = predict(net, p_test );  % 对测试集进行预测

%%  反归一化
T_sim1 = vec2ind(t_sim1');  % 将预测结果转换为类别索引
T_sim2 = vec2ind(t_sim2');  % 将预测结果转换为类别索引

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100;  % 训练集准确率


error2 = sum((T_sim2 == T_test )) / N * 100;  % 测试集准确率

%%  绘制网络分析图
analyzeNetwork(layers);  % 可视化神经网络结构

%%  绘图
% 绘制训练集和测试集预测结果对比
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid

%%  混淆矩阵
if flag_conusion == 1
    % 绘制训练集的混淆矩阵
    figure
    cm = confusionchart(T_train, T_sim1);
    cm.Title = 'Confusion Matrix for Train Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    
    % 绘制测试集的混淆矩阵
    figure
    cm = confusionchart(T_test, T_sim2);
    cm.Title = 'Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end
