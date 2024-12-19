%% 初始化
clear                % 清除工作区变量
close all            % 关闭所有图形窗口
clc                  % 清空命令行窗口
warning off          % 关闭警告信息

%% 导入数据
res = xlsread('数据集.xlsx');  % 从Excel文件中读取数据，假设最后一列为目标变量

%% 数据分析
num_size = 0.7;                              % 设定训练集占数据集的比例（70%训练集，30%测试集）
outdim = 1;                                  % 最后一列为输出（目标变量）
num_samples = size(res, 1);                  % 计算样本个数（数据集中的行数）
res = res(randperm(num_samples), :);         % 随机打乱数据集顺序，以避免数据排序带来的偏差（如果不希望打乱可注释该行）
num_train_s = round(num_size * num_samples); % 计算训练集样本个数（四舍五入）
f_ = size(res, 2) - outdim;                  % 输入特征维度（总列数减去输出维度）

%% 划分训练集和测试集
P_train = res(1:num_train_s, 1:f_)';         % 训练集输入，转置使每列为一个样本 (f_ × M)
T_train = res(1:num_train_s, f_+1:end)';     % 训练集输出，转置使每列为一个样本 (outdim × M)
M = size(P_train, 2);                        % 训练集样本数

P_test = res(num_train_s+1:end, 1:f_)';      % 测试集输入，转置使每列为一个样本 (f_ × N)
T_test = res(num_train_s+1:end, f_+1:end)';  % 测试集输出，转置使每列为一个样本 (outdim × N)
N = size(P_test, 2);                         % 测试集样本数

%% 数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);         % 对训练集输入进行归一化，范围[0,1]
P_test = mapminmax('apply', P_test, ps_input);         % 使用训练集的归一化参数对测试集输入进行归一化

[t_train, ps_output] = mapminmax(T_train, 0, 1);         % 对训练集输出进行归一化，范围[0,1]
t_test = mapminmax('apply', T_test, ps_output);         % 使用训练集的归一化参数对测试集输出进行归一化

%% 节点个数
inputnum  = size(p_train, 1);  % 输入层节点数（特征维度）
hiddennum = 5;                 % 隐藏层节点数
outputnum = size(t_train, 1);  % 输出层节点数（目标维度）

%% 建立网络
net = newff(p_train, t_train, hiddennum);  % 创建BP神经网络，隐藏层节点数为hiddennum

%% 设置训练参数
net.trainParam.epochs     = 1000;      % 设置最大训练次数为1000
net.trainParam.goal       = 1e-6;      % 设置训练目标误差为1e-6
net.trainParam.lr         = 0.01;      % 设置学习率为0.01
net.trainParam.showWindow = 0;         % 关闭训练窗口，避免干扰

%% 参数初始化
c1      = 4.494;       % 学习因子1（个体认知因子）
c2      = 4.494;       % 学习因子2（社会认知因子）
maxgen  =   45;        % 粒子群优化的迭代代数
sizepop =    5;         % 粒子群优化的种群规模
Vmax    =  1.0;        % 最大速度
Vmin    = -1.0;        % 最小速度
popmax  =  1.0;        % 粒子位置最大边界
popmin  = -1.0;        % 粒子位置最小边界

%% 节点总数
% 计算BP网络的所有可优化参数数量（输入权重 + 隐层偏置 + 输出权重 + 输出层偏置）
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

%% 初始化种群和速度
for i = 1:sizepop
    pop(i, :) = rand(1, numsum) * 2 - 1;  % 初始化粒子位置，随机在[-1,1]之间
    V(i, :) = rand(1, numsum) * 2 - 1;    % 初始化粒子速度，随机在[-1,1]之间
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);  % 计算初始适应度
end

%% 个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);   % 找到群体中最优适应度值和对应粒子索引
zbest = pop(bestindex, :);                  % 全局最佳粒子位置
gbest = pop;                                % 个体最佳粒子位置（初始为当前种群位置）
fitnessgbest = fitness;                     % 个体最佳适应度值（初始为当前种群适应度）
BestFit = fitnesszbest;                     % 存储全局最佳适应度值

%% 迭代寻优
for i = 1:maxgen
    for j = 1:sizepop
        % 速度更新公式
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        % 限制速度范围
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % 位置更新公式
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);  % 位置更新步长为0.2
        % 限制位置范围
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % 自适应变异
        pos = unidrnd(numsum);                   % 随机选择一个位置进行变异
        if rand > 0.85                           % 85%的概率触发变异
            pop(j, pos) = rand * 2 - 1;           % 变异为新的随机值[-1,1]
        end
        
        % 适应度值更新
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);
    end
    
    for j = 1:sizepop
        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);            % 更新个体最佳位置
            fitnessgbest(j) = fitness(j);       % 更新个体最佳适应度值
        end
        
        % 群体最优更新
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);                   % 更新全局最佳位置
            fitnesszbest = fitness(j);           % 更新全局最佳适应度值
        end
    end
    
    BestFit = [BestFit, fitnesszbest];            % 记录全局最佳适应度值
end

%% 提取最优初始权值和阈值
% 从全局最佳粒子位置zbest中提取BP网络的权重和偏置
w1 = zbest(1 : inputnum * hiddennum);  % 输入权重向量
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);  % 隐层偏置向量
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum);  % 输出权重向量
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);  % 输出层偏置向量

%% 最优值赋值
% 将提取的权重和偏置赋值给BP神经网络对象
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);  % 输入权重矩阵
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum); % 输出权重矩阵
net.b{1}     = reshape(B1, hiddennum, 1);        % 隐层偏置向量
net.b{2}     = B2';                               % 输出层偏置向量

%% 打开训练窗口 
net.trainParam.showWindow = 1;        % 打开训练窗口，观察训练过程

%% 网络训练
net = train(net, p_train, t_train);    % 使用训练集数据进一步训练BP神经网络

%% 仿真预测
t_sim1 = sim(net, p_train);              % 使用训练集数据进行预测，得到训练集预测结果
t_sim2 = sim(net, p_test );              % 使用测试集数据进行预测，得到测试集预测结果

%% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);  % 将训练集预测结果反归一化，恢复到原始尺度
T_sim2 = mapminmax('reverse', t_sim2, ps_output);  % 将测试集预测结果反归一化，恢复到原始尺度

%% 均方根误差（RMSE）
error1 = sqrt(sum((T_sim1 - T_train).^2, 2)' ./ M);  % 计算训练集的均方根误差（RMSE）
error2 = sqrt(sum((T_sim2 - T_test) .^2, 2)' ./ N);  % 计算测试集的均方根误差（RMSE）

%% 绘图
% 绘制训练集预测结果对比图
figure
plot(1:M, T_train, 'r-*', 1:M, T_sim1, 'b-o', 'LineWidth', 1) % 绘制真实值与预测值对比曲线
legend('真实值', '预测值')                                        % 添加图例
xlabel('预测样本')                                                % 设置X轴标签
ylabel('预测结果')                                                % 设置Y轴标签
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};      % 创建标题字符串
title(string)                                                    % 添加图形标题
xlim([1, M])                                                     % 设置X轴范围
grid                                                             % 显示网格

% 绘制测试集预测结果对比图
figure
plot(1:N, T_test, 'r-*', 1:N, T_sim2, 'b-o', 'LineWidth', 1) % 绘制真实值与预测值对比曲线
legend('真实值', '预测值')                                        % 添加图例
xlabel('预测样本')                                                % 设置X轴标签
ylabel('预测结果')                                                % 设置Y轴标签
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};       % 创建标题字符串
title(string)                                                    % 添加图形标题
xlim([1, N])                                                     % 设置X轴范围
grid                                                             % 显示网格

%% 误差曲线迭代图
figure;
plot(1:length(BestFit), BestFit, 'LineWidth', 1.5);  % 绘制粒子群优化过程中适应度值的变化曲线
xlabel('粒子群迭代次数');                              % 设置X轴标签
ylabel('适应度值');                                    % 设置Y轴标签
xlim([1, length(BestFit)])                             % 设置X轴范围
title('模型迭代误差变化');                              % 设置图形标题
grid on                                                 % 显示网格

%% 相关指标计算
% R²
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;  % 计算训练集的决定系数R²
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;  % 计算测试集的决定系数R²

disp(['训练集数据的R²为：', num2str(R1)])  % 显示训练集的R²
disp(['测试集数据的R²为：', num2str(R2)])  % 显示测试集的R²

% MAE
mae1 = sum(abs(T_sim1 - T_train), 2)' ./ M ;  % 计算训练集的平均绝对误差MAE
mae2 = sum(abs(T_sim2 - T_test ), 2)' ./ N ;  % 计算测试集的平均绝对误差MAE

disp(['训练集数据的MAE为：', num2str(mae1)])  % 显示训练集的MAE
disp(['测试集数据的MAE为：', num2str(mae2)])  % 显示测试集的MAE

% MBE
mbe1 = sum(T_sim1 - T_train, 2)' ./ M ;  % 计算训练集的平均偏差误差MBE
mbe2 = sum(T_sim2 - T_test , 2)' ./ N ;  % 计算测试集的平均偏差误差MBE

disp(['训练集数据的MBE为：', num2str(mbe1)])  % 显示训练集的MBE
disp(['测试集数据的MBE为：', num2str(mbe2)])  % 显示测试集的MBE

% MAPE
mape1 = sum(abs((T_sim1 - T_train)./T_train)) ./ M ;  % 计算训练集的平均绝对百分比误差MAPE
mape2 = sum(abs((T_sim2 - T_test )./T_test )) ./ N ;  % 计算测试集的平均绝对百分比误差MAPE

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
