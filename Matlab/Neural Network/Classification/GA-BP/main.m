%% 初始化
clear         % 清除工作区中的所有变量
close all     % 关闭所有打开的图形窗口
clc           % 清除命令窗口的内容
warning off   % 关闭所有警告信息

%% 导入数据
res = xlsread('数据集.xlsx'); % 从 Excel 文件中读取数据，存储在变量 res 中

%% 添加路径
addpath('goat\') % 添加包含遗传算法相关函数的文件夹路径

%% 分析数据
num_class = length(unique(res(:, end)));  % 计算类别数，假设最后一列为类别标签
num_res = size(res, 1);                   % 计算样本总数，每一行代表一个样本
num_size = 0.7;                           % 设置训练集占数据集的比例为 70%
res = res(randperm(num_res), :);          % 随机打乱数据集顺序，提高模型泛化能力
flag_conusion = 1;                        % 设置标志位为 1，启用混淆矩阵绘制（要求 MATLAB 2018 及以上版本）

%% 设置变量存储数据
P_train = []; P_test = []; % 初始化训练集和测试集的输入特征
T_train = []; T_test = []; % 初始化训练集和测试集的目标输出

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
P_train = P_train'; P_test = P_test'; % 转置训练集和测试集输入特征，使每列代表一个样本
T_train = T_train'; T_test = T_test'; % 转置训练集和测试集目标输出

%% 得到训练集和测试样本个数
M = size(P_train, 2); % 训练集样本数量
N = size(P_test , 2); % 测试集样本数量

%% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);               % 将训练集输入特征归一化到 [0,1] 范围
p_test  = mapminmax('apply', P_test, ps_input);               % 使用相同的归一化参数处理测试集输入特征
t_train = ind2vec(T_train);                                   % 将训练集目标输出转换为独热编码
t_test  = ind2vec(T_test );                                   % 将测试集目标输出转换为独热编码

%% 建立模型
S1 = 5;           % 设置隐藏层节点个数为 5
net = newff(p_train, t_train, S1); % 创建前馈神经网络，使用新建的 BP 网络

%% 设置参数
net.trainParam.epochs = 1000;        % 设置最大训练迭代次数为 1000 次
net.trainParam.goal   = 1e-6;        % 设置训练目标误差为 1e-6
net.trainParam.lr     = 0.01;        % 设置学习率为 0.01

%% 设置优化参数
gen = 50;                       % 设置遗传算法的最大代数为 50
pop_num = 5;                    % 设置种群规模为 5
S = size(p_train, 1) * S1 + S1 * size(t_train, 1) + S1 + size(t_train, 1);
                                % 计算优化参数个数：输入权重 + 输出权重 + 偏置
bounds = ones(S, 1) * [-1, 1];  % 设置优化变量的边界为 [-1, 1] 之间

%% 初始化种群
prec = [1e-6, 1];               % 设置精度和编码方式：epslin 为 1e-6，实数编码
normGeomSelect = 0.09;          % 设置选择函数的参数
arithXover = 2;                 % 设置交叉函数的参数
nonUnifMutation = [2 gen 3];    % 设置变异函数的参数

initPop = initializega(pop_num, bounds, 'gabpEval', [], prec);  
                                % 初始化遗传算法的种群

%% 优化算法
[Bestpop, endPop, bPop, trace] = ga(bounds, 'gabpEval', [], initPop, [prec, 0], 'maxGenTerm', gen,...
                           'normGeomSelect', normGeomSelect, 'arithXover', arithXover, ...
                           'nonUnifMutation', nonUnifMutation);
                                % 运行遗传算法，优化神经网络参数

%% 获取最优参数
[val, W1, B1, W2, B2] = gadecod(Bestpop); % 解码最优参数，得到权重和偏置

%% 参数赋值
net.IW{1, 1} = W1; % 设置输入层到隐藏层的权重
net.LW{2, 1} = W2; % 设置隐藏层到输出层的权重
net.b{1}     = B1; % 设置隐藏层的偏置
net.b{2}     = B2; % 设置输出层的偏置

%% 模型训练
net.trainParam.showWindow = 1;       % 打开训练窗口
net = train(net, p_train, t_train);  % 使用 BP 算法训练神经网络

%% 仿真测试
t_sim1 = sim(net, p_train); % 使用训练数据进行仿真测试，得到训练集的预测输出
t_sim2 = sim(net, p_test ); % 使用测试数据进行仿真测试，得到测试集的预测输出

%% 数据反归一化
T_sim1 = vec2ind(t_sim1); % 将训练集预测输出的独热编码转换为类别索引
T_sim2 = vec2ind(t_sim2); % 将测试集预测输出的独热编码转换为类别索引

%% 性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ; % 计算训练集的分类准确率（百分比）
error2 = sum((T_sim2 == T_test )) / N * 100 ; % 计算测试集的分类准确率（百分比）

%% 数据排序
[T_train, index_1] = sort(T_train); % 对训练集真实标签进行排序，并获取排序索引
[T_test , index_2] = sort(T_test ); % 对测试集真实标签进行排序，并获取排序索引

T_sim1 = T_sim1(index_1); % 根据排序索引重新排列训练集预测结果
T_sim2 = T_sim2(index_2); % 根据排序索引重新排列测试集预测结果

%% 优化迭代曲线
figure
plot(trace(:, 1), 1 ./ trace(:, 2), 'LineWidth', 1.5); % 绘制适应度值随迭代次数变化的曲线
xlabel('迭代次数');                                      % 设置 X 轴标签
ylabel('适应度值');                                      % 设置 Y 轴标签
title({'适应度变化曲线'});                              % 设置图形标题
grid on                                                  % 显示网格

%% 绘图
% 绘制训练集真实值与预测值对比图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')                 % 添加图例
xlabel('预测样本')                       % 设置 X 轴标签
ylabel('预测结果')                       % 设置 Y 轴标签
title({'训练集预测结果对比'; ['准确率=' num2str(error1) '%']}) % 设置图形标题，显示准确率
grid on                                  % 显示网格

% 绘制测试集真实值与预测值对比图
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')                 % 添加图例
xlabel('预测样本')                       % 设置 X 轴标签
ylabel('预测结果')                       % 设置 Y 轴标签
title({'测试集预测结果对比'; ['准确率=' num2str(error2) '%']}) % 设置图形标题，显示准确率
grid on                                  % 显示网格

%% 混淆矩阵
% 绘制训练集的混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = '训练集混淆矩阵';                     % 设置混淆矩阵标题
cm.ColumnSummary = 'column-normalized'; % 列归一化显示
cm.RowSummary = 'row-normalized';       % 行归一化显示

% 绘制测试集的混淆矩阵
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = '测试集混淆矩阵';                     % 设置混淆矩阵标题
cm.ColumnSummary = 'column-normalized'; % 列归一化显示
cm.RowSummary = 'row-normalized';       % 行归一化显示
