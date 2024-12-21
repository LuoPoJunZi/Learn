%% 清空环境变量
warning off             % 关闭所有警告信息，避免运行过程中显示不必要的警告
close all               % 关闭所有打开的图形窗口，确保绘图环境的干净
clear                   % 清除工作区中的所有变量，确保没有残留变量影响结果
clc                     % 清空命令行窗口，提升可读性

%% 导入数据（时间序列的单列数据）
result = xlsread('数据集.xlsx');  % 从Excel文件中读取时间序列数据，假设数据为单列
% result 变量存储了读取的时间序列数据，数据应按照时间顺序排列

%% 数据分析
num_samples = length(result);  % 计算时间序列数据的样本数量（数据点数）
kim = 15;                      % 设定延时步长（lag），即使用15个历史数据点作为输入特征
zim =  1;                      % 设定预测步长（forecast step），即预测当前点之后的1个时间点

%% 构造数据集
for i = 1:num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i:i + kim - 1), 1, kim), result(i + kim + zim - 1)];
end
% 循环遍历时间序列数据，构建输入特征和对应的目标变量
% 每一行res包含15个历史数据点和1个未来数据点

%% 数据集分析
outdim = 1;                                  % 设定数据集的最后一列为输出（目标变量）
num_size = 0.7;                              % 设定训练集占数据集的比例（70%训练集，30%测试集）
num_train_s = round(num_size * num_samples); % 计算训练集样本个数，通过四舍五入确定
f_ = size(res, 2) - outdim;                  % 计算输入特征的维度，即总列数减去输出维度

%% 划分训练集和测试集
P_train = res(1:num_train_s, 1:f_)';         % 训练集输入特征，转置使每列为一个样本 (f_ × M)
T_train = res(1:num_train_s, f_ + 1:end)';   % 训练集输出目标变量，转置使每列为一个样本 (outdim × M)
M = size(P_train, 2);                        % 获取训练集的样本数量

P_test = res(num_train_s + 1:end, 1:f_)';    % 测试集输入特征，转置使每列为一个样本 (f_ × N)
T_test = res(num_train_s + 1:end, f_ + 1:end)';% 测试集输出目标变量，转置使每列为一个样本 (outdim × N)
N = size(P_test, 2);                         % 获取测试集的样本数量

%% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);         % 对训练集输入特征进行归一化，范围[0,1]
p_test = mapminmax('apply', P_test, ps_input);           % 使用训练集的归一化参数对测试集输入特征进行归一化

[t_train, ps_output] = mapminmax(T_train, 0, 1);         % 对训练集输出目标变量进行归一化，范围[0,1]
t_test = mapminmax('apply', T_test, ps_output);           % 使用训练集的归一化参数对测试集输出目标变量进行归一化

%% 转置以适应模型
p_train = p_train'; p_test = p_test';                     % 转置输入特征矩阵，使每行为一个样本
t_train = t_train'; t_test = t_test';                     % 转置目标变量矩阵，使每行为一个样本

%% 创建模型
c = 4.0;    % 惩罚因子，控制误分类的惩罚程度
g = 0.8;    % 径向基函数参数，影响核函数的宽度
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
% 构建SVM命令参数：
% -t 2 表示使用径向基函数（RBF）核
% -c 设置惩罚因子
% -g 设置RBF核函数的gamma参数
% -s 3 表示epsilon-SVR回归
% -p 0.01 设置epsilon的值

model = svmtrain(t_train, p_train, cmd);  % 使用svmtrain函数训练SVM回归模型
% t_train 为目标变量，p_train 为输入特征，cmd 为命令参数

%% 仿真预测
[t_sim1, error_1] = svmpredict(t_train, p_train, model);  % 使用训练集数据进行预测，得到训练集预测结果和误差
[t_sim2, error_2] = svmpredict(t_test , p_test , model);  % 使用测试集数据进行预测，得到测试集预测结果和误差

%% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);  % 将训练集预测结果反归一化，恢复到原始尺度
T_sim2 = mapminmax('reverse', t_sim2, ps_output);  % 将测试集预测结果反归一化，恢复到原始尺度

%% 均方根误差（RMSE）
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);  % 计算训练集的均方根误差（RMSE）
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);  % 计算测试集的均方根误差（RMSE）

%% 绘图
% 绘制训练集预测结果对比图
figure
plot(1:M, T_train, 'r-', 1:M, T_sim1, 'b-', 'LineWidth', 1) % 绘制训练集真实值与预测值的对比曲线，红色实线为真实值，蓝色实线为预测值
legend('真实值', '预测值')                                        % 添加图例，区分真实值和预测值
xlabel('预测样本')                                                % 设置X轴标签为“预测样本”
ylabel('预测结果')                                                % 设置Y轴标签为“预测结果”
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};      % 创建标题字符串，包括RMSE值
title(string)                                                    % 添加图形标题
xlim([1, M])                                                     % 设置X轴显示范围为[1, M]
grid                                                             % 显示网格，提升图形的可读性

% 绘制测试集预测结果对比图
figure
plot(1:N, T_test, 'r-', 1:N, T_sim2, 'b-', 'LineWidth', 1)   % 绘制测试集真实值与预测值的对比曲线，红色实线为真实值，蓝色实线为预测值
legend('真实值', '预测值')                                        % 添加图例，区分真实值和预测值
xlabel('预测样本')                                                % 设置X轴标签为“预测样本”
ylabel('预测结果')                                                % 设置Y轴标签为“预测结果”
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};       % 创建标题字符串，包括RMSE值
title(string)                                                    % 添加图形标题
xlim([1, N])                                                     % 设置X轴显示范围为[1, N]
grid                                                             % 显示网格，提升图形的可读性

%% 相关指标计算

% 决定系数（R²）
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;  % 计算训练集的决定系数R²
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;    % 计算测试集的决定系数R²

disp(['训练集数据的R2为：', num2str(R1)])  % 显示训练集的R²
disp(['测试集数据的R2为：', num2str(R2)])  % 显示测试集的R²

% 平均绝对误差（MAE）
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;  % 计算训练集的平均绝对误差MAE
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;  % 计算测试集的平均绝对误差MAE

disp(['训练集数据的MAE为：', num2str(mae1)])  % 显示训练集的MAE
disp(['测试集数据的MAE为：', num2str(mae2)])  % 显示测试集的MAE

% 平均偏差误差（MBE）
mbe1 = sum(T_sim1' - T_train) ./ M ;  % 计算训练集的平均偏差误差MBE
mbe2 = sum(T_sim2' - T_test ) ./ N ;  % 计算测试集的平均偏差误差MBE

disp(['训练集数据的MBE为：', num2str(mbe1)])  % 显示训练集的MBE
disp(['测试集数据的MBE为：', num2str(mbe2)])  % 显示测试集的MBE

% 平均绝对百分比误差（MAPE）
mape1 = sum(abs((T_sim1' - T_train)./T_train)) ./ M ;  % 计算训练集的平均绝对百分比误差MAPE
mape2 = sum(abs((T_sim2' - T_test )./T_test )) ./ N ;  % 计算测试集的平均绝对百分比误差MAPE

disp(['训练集数据的MAPE为：', num2str(mape1)])  % 显示训练集的MAPE
disp(['测试集数据的MAPE为：', num2str(mape2)])  % 显示测试集的MAPE

% 均方根误差（RMSE）
disp(['训练集数据的RMSE为：', num2str(error1)])  % 显示训练集的RMSE
disp(['测试集数据的RMSE为：', num2str(error2)])  % 显示测试集的RMSE

%% 绘制散点图
sz = 25;       % 设置散点的大小为25
c = 'b';       % 设置散点的颜色为蓝色

% 绘制训练集散点图
figure
scatter(T_train, T_sim1, sz, c)              % 绘制训练集真实值与预测值的散点图，蓝色散点表示预测结果
hold on                                       % 保持当前图形，允许在同一图形上绘制多条曲线
plot(xlim, ylim, '--k')                      % 绘制理想预测线（真实值等于预测值的对角线），使用黑色虚线表示
xlabel('训练集真实值');                        % 设置X轴标签为“训练集真实值”
ylabel('训练集预测值');                        % 设置Y轴标签为“训练集预测值”
xlim([min(T_train) max(T_train)])             % 设置X轴的显示范围为[最小真实值, 最大真实值]
ylim([min(T_sim1) max(T_sim1)])               % 设置Y轴的显示范围为[最小预测值, 最大预测值]
title('训练集预测值 vs. 训练集真实值')            % 设置图形的标题为“训练集预测值 vs. 训练集真实值”

% 绘制测试集散点图
figure
scatter(T_test, T_sim2, sz, c)               % 绘制测试集真实值与预测值的散点图，蓝色散点表示预测结果
hold on                                       % 保持当前图形，允许在同一图形上绘制多条曲线
plot(xlim, ylim, '--k')                      % 绘制理想预测线（真实值等于预测值的对角线），使用黑色虚线表示
xlabel('测试集真实值');                         % 设置X轴标签为“测试集真实值”
ylabel('测试集预测值');                         % 设置Y轴标签为“测试集预测值”
xlim([min(T_test) max(T_test)])                % 设置X轴的显示范围为[最小真实值, 最大真实值]
ylim([min(T_sim2) max(T_sim2)])                % 设置Y轴的显示范围为[最小预测值, 最大预测值]
title('测试集预测值 vs. 测试集真实值')             % 设置图形的标题为“测试集预测值 vs. 测试集真实值”
