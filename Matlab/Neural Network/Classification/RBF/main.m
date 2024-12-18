%% 初始化
clear                % 清除工作区变量
close all            % 关闭所有图形窗口
clc                  % 清空命令行窗口
warning off          % 关闭警告信息

%% 导入数据
res = xlsread('数据集.xlsx');  % 从Excel文件中读取数据，假设最后一列为类别标签

%% 分析数据
num_class = length(unique(res(:, end)));  % 计算类别数（假设最后一列为类别标签）
num_res = size(res, 1);                   % 计算样本数（数据集中的行数）
num_size = 0.7;                           % 设定训练集占数据集的比例（70%训练集，30%测试集）
res = res(randperm(num_res), :);          % 随机打乱数据集顺序，以避免数据排序带来的偏差
flag_conusion = 1;                        % 设置标志位为1，表示需要绘制混淆矩阵（要求MATLAB 2018及以上版本）

%% 设置变量存储数据
P_train = []; P_test = [];    % 初始化训练集和测试集的输入数据矩阵
T_train = []; T_test = [];    % 初始化训练集和测试集的输出数据矩阵

%% 划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 提取当前类别的所有样本
    mid_size = size(mid_res, 1);                    % 计算当前类别的样本数
    mid_tiran = round(num_size * mid_size);         % 计算当前类别训练样本的数量（四舍五入）

    % 划分训练集输入和输出
    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 将当前类别的训练集输入添加到P_train
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 将当前类别的训练集输出添加到T_train

    % 划分测试集输入和输出
    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 将当前类别的测试集输入添加到P_test
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 将当前类别的测试集输出添加到T_test
end

%% 数据转置
P_train = P_train';  % 转置训练集输入，使每列为一个样本
P_test = P_test';    % 转置测试集输入
T_train = T_train';  % 转置训练集输出
T_test = T_test';    % 转置测试集输出

%% 得到训练集和测试样本个数
M = size(P_train, 2);  % 训练集样本数
N = size(P_test , 2);  % 测试集样本数

%% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);          % 对训练集输入进行归一化，范围[0,1]
p_test  = mapminmax('apply', P_test, ps_input);         % 使用训练集的归一化参数对测试集输入进行归一化
t_train = ind2vec(T_train);                             % 将训练集输出转换为向量编码（类别编码）
t_test  = ind2vec(T_test );                             % 将测试集输出转换为向量编码

%% 创建网络
rbf_spread = 100;                           % 设置径向基函数的扩展速度（spread参数）
net = newrbe(p_train, t_train, rbf_spread); % 使用newrbe函数创建RBF神经网络，并训练输出层权重

%% 仿真测试
t_sim1 = sim(net, p_train);          % 使用训练集进行仿真预测
t_sim2 = sim(net, p_test );          % 使用测试集进行仿真预测

%% 数据反归一化
T_sim1 = vec2ind(t_sim1);             % 将训练集预测结果转换为类别索引
T_sim2 = vec2ind(t_sim2);             % 将测试集预测结果转换为类别索引

%% 性能评价
error1 = sum((T_sim1 == T_train)) / M * 100;  % 计算训练集的分类准确率
error2 = sum((T_sim2 == T_test )) / N * 100;  % 计算测试集的分类准确率

%% 网络结构可视化
view(net)                             % 可视化RBF神经网络的结构

%% 数据排序
[T_train, index_1] = sort(T_train);     % 对训练集真实标签进行排序，获取排序索引
[T_test , index_2] = sort(T_test );     % 对测试集真实标签进行排序，获取排序索引

T_sim1 = T_sim1(index_1);                % 按排序索引调整训练集预测结果
T_sim2 = T_sim2(index_2);                % 按排序索引调整测试集预测结果

%% 绘图
% 绘制训练集预测结果对比图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid

% 绘制测试集预测结果对比图
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid

%% 混淆矩阵
if flag_conusion
    % 绘制训练集混淆矩阵
    figure
    cm = confusionchart(T_train, T_sim1);
    cm.Title = '训练集混淆矩阵';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
        
    % 绘制测试集混淆矩阵
    figure
    cm = confusionchart(T_test, T_sim2);
    cm.Title = '测试集混淆矩阵';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end
