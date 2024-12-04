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