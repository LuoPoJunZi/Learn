%% 初始化
clear                % 清除工作区变量
close all            % 关闭所有图形窗口
clc                  % 清空命令行窗口
warning off          % 关闭警告信息

%% 导入数据
res = xlsread('数据集.xlsx');  % 从Excel文件中读取数据，假设最后一列为类别标签

%% 数据分析
num_size = 0.7;                              % 设定训练集占数据集的比例（70%训练集，30%测试集）
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 计算样本个数（数据集中的行数）
res = res(randperm(num_samples), :);         % 随机打乱数据集顺序，以避免数据排序带来的偏差（如果不需要打乱可注释该行）
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
[p_train, ps_input] = mapminmax(P_train, 0, 1);          % 对训练集输入进行归一化，范围[0,1]
p_test = mapminmax('apply', P_test, ps_input );         % 使用训练集的归一化参数对测试集输入进行归一化
t_train = T_train;                                      % 保持训练集输出不变
t_test  = T_test ;                                      % 保持测试集输出不变

%% 转置以适应模型
p_train = p_train'; p_test = p_test';                   % 转置输入数据，使每列为一个样本
t_train = t_train'; t_test = t_test';                   % 转置输出数据，使每列为一个样本

%% 训练模型
trees = 50;                                       % 决策树数目（可调节）
leaf  = 1;                                        % 最小叶子节点数（防止过拟合）
OOBPrediction = 'on';                             % 打开袋外误差预测
OOBPredictorImportance = 'on';                    % 计算特征重要性
Method = 'classification';                        % 设定方法为分类
% 使用TreeBagger函数创建随机森林模型，并设置相关参数
net = TreeBagger(trees, p_train, t_train, ...
    'OOBPredictorImportance', OOBPredictorImportance, ...
    'Method', Method, ...
    'OOBPrediction', OOBPrediction, ...
    'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  % 获取特征重要性

%% 仿真测试
t_sim1 = predict(net, p_train);          % 使用训练集进行预测
t_sim2 = predict(net, p_test );          % 使用测试集进行预测

%% 格式转换
T_sim1 = str2double(t_sim1);             % 将训练集预测结果转换为数值型
T_sim2 = str2double(t_sim2);             % 将测试集预测结果转换为数值型

%% 性能评价
error1 = sum((T_sim1' == T_train)) / M * 100 ;  % 计算训练集的分类准确率
error2 = sum((T_sim2' == T_test )) / N * 100 ;  % 计算测试集的分类准确率

%% 绘制误差曲线
figure
plot(1: trees, oobError(net), 'b-', 'LineWidth', 1) % 绘制袋外误差随决策树数目的变化曲线
legend('误差曲线')                                   % 添加图例
xlabel('决策树数目')                                  % X轴标签
ylabel('误差')                                       % Y轴标签
xlim([1, trees])                                     % 设置X轴范围
grid                                                % 显示网格

%% 绘制特征重要性
figure
bar(importance)                                     % 绘制特征重要性柱状图
legend('重要性')                                     % 添加图例
xlabel('特征')                                       % X轴标签
ylabel('重要性')                                     % Y轴标签

%% 数据排序
[T_train, index_1] = sort(T_train);     % 对训练集真实标签进行排序，获取排序索引
[T_test , index_2] = sort(T_test );     % 对测试集真实标签进行排序，获取排序索引

T_sim1 = T_sim1(index_1);                % 按排序索引调整训练集预测结果
T_sim2 = T_sim2(index_2);                % 按排序索引调整测试集预测结果

%% 绘图
% 绘制训练集预测结果对比图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1) % 绘制真实值与预测值对比
legend('真实值', '预测值')                                        % 添加图例
xlabel('预测样本')                                                % X轴标签
ylabel('预测结果')                                                % Y轴标签
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};  % 创建标题字符串
title(string)                                                    % 添加标题
grid                                                             % 显示网格

% 绘制测试集预测结果对比图
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1) % 绘制真实值与预测值对比
legend('真实值', '预测值')                                        % 添加图例
xlabel('预测样本')                                                % X轴标签
ylabel('预测结果')                                                % Y轴标签
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};   % 创建标题字符串
title(string)                                                    % 添加标题
grid                                                             % 显示网格

%% 混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);  % 创建训练集的混淆矩阵图
cm.Title = '训练集混淆矩阵';               % 设置混淆矩阵标题
cm.ColumnSummary = 'column-normalized';      % 设置列摘要为归一化
cm.RowSummary = 'row-normalized';            % 设置行摘要为归一化

figure
cm = confusionchart(T_test, T_sim2);    % 创建测试集的混淆矩阵图
cm.Title = '测试集混淆矩阵';               % 设置混淆矩阵标题
cm.ColumnSummary = 'column-normalized';      % 设置列摘要为归一化
cm.RowSummary = 'row-normalized';            % 设置行摘要为归一化
