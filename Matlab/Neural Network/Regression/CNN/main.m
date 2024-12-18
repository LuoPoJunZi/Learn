%% 初始化
clear                % 清除工作区变量
close all            % 关闭所有图形窗口
clc                  % 清空命令行窗口
warning off          % 关闭警告信息

%% 导入数据
res = xlsread('数据集.xlsx');  % 从Excel文件中读取数据，假设最后一列为目标变量

%% 数据分析
num_size = 0.7;                              % 设定训练集占数据集的比例（70%训练集，30%测试集）
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 计算样本个数（数据集中的行数）
res = res(randperm(num_samples), :);         % 随机打乱数据集顺序，以避免数据排序带来的偏差（如果不希望打乱可注释该行）
num_train_s = round(num_size * num_samples); % 计算训练集样本个数（四舍五入）
f_ = size(res, 2) - outdim;                  % 输入特征维度（总列数减去输出维度）

%% 划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';       % 训练集输入，转置使每列为一个样本
T_train = res(1: num_train_s, f_ + 1: end)'; % 训练集输出，转置使每列为一个样本
M = size(P_train, 2);                        % 训练集样本数

P_test = res(num_train_s + 1: end, 1: f_)';   % 测试集输入，转置使每列为一个样本
T_test = res(num_train_s + 1: end, f_ + 1: end)'; % 测试集输出，转置使每列为一个样本
N = size(P_test, 2);                          % 测试集样本数

%% 数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);          % 对训练集输入进行归一化，范围[0,1]
P_test = mapminmax('apply', P_test, ps_input );         % 使用训练集的归一化参数对测试集输入进行归一化

[t_train, ps_output] = mapminmax(T_train, 0, 1);          % 对训练集输出进行归一化，范围[0,1]
t_test = mapminmax('apply', T_test, ps_output );         % 使用训练集的归一化参数对测试集输出进行归一化

%% 数据平铺
% 将数据平铺成4维数据，以适应MATLAB的CNN输入格式
% 输入格式为 [高度, 宽度, 通道数, 样本数]
% 这里将数据平铺成 [特征维度, 1, 1, 样本数] 的格式
p_train = double(reshape(P_train, f_, 1, 1, M));        % 将训练集输入平铺成4维数据
p_test  = double(reshape(P_test , f_, 1, 1, N));        % 将测试集输入平铺成4维数据
t_train = double(t_train)';                             % 将训练集输出转置为列向量
t_test  = double(t_test )';                             % 将测试集输出转置为列向量

%% 构造网络结构
% ----------------------  修改模型结构时需对应修改fical.m中的模型结构  --------------------------

layers = [
    imageInputLayer([f_, 1, 1])                        % 输入层 输入数据规模 [特征维度, 1, 1]
    
    convolution2dLayer([3, 1], 16, 'Padding', 'same')  % 卷积层 卷积核大小 3x1，生成16张特征图，使用'same'填充保持尺寸
    batchNormalizationLayer                            % 批归一化层 规范化数据，加快训练速度并稳定性
    reluLayer                                          % ReLU激活层 引入非线性
    
    convolution2dLayer([3, 1], 32, 'Padding', 'same')  % 卷积层 卷积核大小 3x1，生成32张特征图，使用'same'填充保持尺寸
    batchNormalizationLayer                            % 批归一化层
    reluLayer                                          % ReLU激活层
    
    dropoutLayer(0.2)                                  % Dropout层 随机丢弃20%的神经元，防止过拟合
    fullyConnectedLayer(outdim)                        % 全连接层 输出维度与目标变量相同
    regressionLayer];                                  % 回归层 使用回归损失函数

%% 参数设置
options = trainingOptions('adam', ...      % 选择Adam优化算法
    'MaxEpochs', 500, ...                  % 设置最大训练次数为500
    'InitialLearnRate', 1e-3, ...          % 设置初始学习率为0.001
    'L2Regularization', 1e-4, ...          % 设置L2正则化参数为0.0001，防止过拟合
    'LearnRateSchedule', 'piecewise', ...  % 设置学习率下降策略为分段
    'LearnRateDropFactor', 0.1, ...        % 设置学习率下降因子为0.1
    'LearnRateDropPeriod', 400, ...        % 设置在第400个epoch后学习率下降
    'Shuffle', 'every-epoch', ...          % 每个epoch后打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证提前停止策略
    'Plots', 'training-progress', ...      % 显示训练过程中的损失曲线
    'Verbose', false);                     % 关闭命令行中的训练过程输出

%% 训练网络
net = trainNetwork(p_train, t_train, layers, options);  % 使用训练集数据训练CNN网络

%% 仿真验证
t_sim1 = predict(net, p_train);          % 使用训练集数据进行预测，得到训练集的预测结果
t_sim2 = predict(net, p_test );          % 使用测试集数据进行预测，得到测试集的预测结果

%% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);  % 将训练集预测结果反归一化，恢复到原始尺度
T_sim2 = mapminmax('reverse', t_sim2, ps_output);  % 将测试集预测结果反归一化，恢复到原始尺度

%% 均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);  % 计算训练集的均方根误差（RMSE）
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);  % 计算测试集的均方根误差（RMSE）

%% 绘制网络分析图
analyzeNetwork(layers)  % 可视化网络结构和各层的参数

%% 绘图
% 绘制训练集预测结果对比图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1) % 绘制真实值与预测值对比曲线
legend('真实值', '预测值')                                        % 添加图例
xlabel('预测样本')                                                % 设置X轴标签
ylabel('预测结果')                                                % 设置Y轴标签
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};      % 创建标题字符串
title(string)                                                    % 添加图形标题
xlim([1, M])                                                     % 设置X轴范围
grid                                                             % 显示网格

% 绘制测试集预测结果对比图
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1) % 绘制真实值与预测值对比曲线
legend('真实值', '预测值')                                        % 添加图例
xlabel('预测样本')                                                % 设置X轴标签
ylabel('预测结果')                                                % 设置Y轴标签
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};       % 创建标题字符串
title(string)                                                    % 添加图形标题
xlim([1, N])                                                     % 设置X轴范围
grid                                                             % 显示网格

%% 相关指标计算
% R²
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;  % 计算训练集的决定系数R²
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;  % 计算测试集的决定系数R²

disp(['训练集数据的R2为：', num2str(R1)])  % 显示训练集的R²
disp(['测试集数据的R2为：', num2str(R2)])  % 显示测试集的R²

% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;  % 计算训练集的平均绝对误差MAE
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;  % 计算测试集的平均绝对误差MAE

disp(['训练集数据的MAE为：', num2str(mae1)])  % 显示训练集的MAE
disp(['测试集数据的MAE为：', num2str(mae2)])  % 显示测试集的MAE

% MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;  % 计算训练集的平均偏差误差MBE
mbe2 = sum(T_sim2' - T_test ) ./ N ;  % 计算测试集的平均偏差误差MBE

disp(['训练集数据的MBE为：', num2str(mbe1)])  % 显示训练集的MBE
disp(['测试集数据的MBE为：', num2str(mbe2)])  % 显示测试集的MBE

% MAPE
mape1 = sum(abs((T_sim1' - T_train)./T_train)) ./ M ;  % 计算训练集的平均绝对百分比误差MAPE
mape2 = sum(abs((T_sim2' - T_test )./T_test )) ./ N ;  % 计算测试集的平均绝对百分比误差MAPE

disp(['训练集数据的MAPE为：', num2str(mape1)])  % 显示训练集的MAPE
disp(['测试集数据的MAPE为：', num2str(mape2)])  % 显示测试集的MAPE

% RMSE
disp(['训练集数据的RMSE为：', num2str(error1)])  % 显示训练集的RMSE
disp(['测试集数据的RMSE为：', num2str(error2)])  % 显示测试集的RMSE

%% 绘制散点图
sz = 25;       % 设置散点大小
c = 'b';       % 设置散点颜色为蓝色

% 绘制训练集散点图
figure
scatter(T_train, T_sim1, sz, c)              % 绘制训练集真实值与预测值的散点图
hold on                                       % 保持图形
plot(xlim, ylim, '--k')                       % 绘制理想预测线（真实值等于预测值的对角线）
xlabel('训练集真实值');                        % 设置X轴标签
ylabel('训练集预测值');                        % 设置Y轴标签
xlim([min(T_train) max(T_train)])              % 设置X轴范围
ylim([min(T_sim1) max(T_sim1)])                % 设置Y轴范围
title('训练集预测值 vs. 训练集真实值')            % 设置图形标题

% 绘制测试集散点图
figure
scatter(T_test, T_sim2, sz, c)               % 绘制测试集真实值与预测值的散点图
hold on                                       % 保持图形
plot(xlim, ylim, '--k')                       % 绘制理想预测线（真实值等于预测值的对角线）
xlabel('测试集真实值');                         % 设置X轴标签
ylabel('测试集预测值');                         % 设置Y轴标签
xlim([min(T_test) max(T_test)])                 % 设置X轴范围
ylim([min(T_sim2) max(T_sim2)])                 % 设置Y轴范围
title('测试集预测值 vs. 测试集真实值')             % 设置图形标题
