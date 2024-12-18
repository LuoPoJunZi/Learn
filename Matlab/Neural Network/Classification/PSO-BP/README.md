### 代码简介

该MATLAB代码实现了基于粒子群优化（PSO）和反向传播（BP）神经网络的分类算法，简称“PSO-BP分类”。其主要流程如下：

1. **数据预处理**：
   - 导入数据集，并随机打乱数据顺序。
   - 将数据集划分为训练集和测试集。
   - 对数据进行归一化处理，以提高训练效果。

2. **神经网络构建**：
   - 使用BP神经网络作为基础模型。
   - 设置输入层、隐藏层和输出层的节点数。

3. **粒子群优化（PSO）**：
   - 初始化粒子群，包括位置和速度。
   - 通过PSO算法优化神经网络的权重和阈值，以最小化分类错误率。
   - 迭代更新粒子的位置和速度，寻找全局最优解。

4. **模型训练与测试**：
   - 使用优化后的权重和阈值训练神经网络。
   - 对训练集和测试集进行预测，并计算分类准确率。
   - 绘制预测结果对比图、误差迭代曲线以及混淆矩阵，以评估模型性能。

以下是添加了详细中文注释的`fun.m`和`main.m`代码。

---

### `fun.m` 文件代码（添加中文注释）

```matlab
function error = fun(pop, hiddennum, net, p_train, t_train)
% FUN 计算粒子对应的误差值，用于PSO优化
% 输入：
%   pop       - 当前粒子的位置（神经网络的权重和阈值）
%   hiddennum - 隐藏层节点数
%   net       - 神经网络对象
%   p_train   - 训练集输入数据
%   t_train   - 训练集目标输出
% 输出：
%   error     - 当前粒子的适应度值（分类错误率）

    %% 节点个数
    inputnum  = size(p_train, 1);  % 输入层节点数
    outputnum = size(t_train, 1);  % 输出层节点数
    
    %% 提取权值和阈值
    % 从粒子位置向量中提取输入层到隐藏层的权重w1
    w1 = pop(1 : inputnum * hiddennum);
    % 提取隐藏层的阈值B1
    B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
    % 提取隐藏层到输出层的权重w2
    w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum);
    % 提取输出层的阈值B2
    B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);
    
    %% 网络赋值
    % 将提取的权重和阈值赋值给神经网络
    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );    % 输入层到隐藏层的权重
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);    % 隐藏层到输出层的权重
    net.b{1}     = reshape(B1, hiddennum, 1);           % 隐藏层的阈值
    net.b{2}     = B2';                                 % 输出层的阈值
    
    %% 网络训练
    net = train(net, p_train, t_train);  % 使用训练集训练网络
    
    %% 仿真测试
    t_sim1 = sim(net, p_train);          % 使用训练集进行仿真预测
    
    %% 反归一化
    T_sim1  = vec2ind(t_sim1 );          % 将网络输出转换为类别索引
    T_train = vec2ind(t_train);           % 将训练集目标输出转换为类别索引
    
    %% 适应度值
    error = 1 - sum(T_sim1 == T_train) / length(T_sim1);  % 计算分类错误率
end
```

---

### `main.m` 文件代码（添加中文注释）

```matlab
%% 初始化
clear                % 清除工作区变量
close all            % 关闭所有图形窗口
clc                  % 清空命令行窗口
warning off          % 关闭警告信息

%% 导入数据
res = xlsread('数据集.xlsx');  % 从Excel文件中读取数据

%% 分析数据
num_class = length(unique(res(:, end)));  % 类别数（假设最后一列为类别标签）
num_res = size(res, 1);                   % 样本数（数据集中的行数）
num_size = 0.7;                           % 训练集占比（70%作为训练集）
res = res(randperm(num_res), :);          % 随机打乱数据集顺序（如果不需要打乱可注释该行）
flag_conusion = 1;                        % 混淆矩阵标志位，1表示显示

%% 设置变量存储数据
P_train = []; P_test = [];    % 输入数据的训练集和测试集
T_train = []; T_test = [];    % 输出数据的训练集和测试集

%% 划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 提取当前类别的所有样本
    mid_size = size(mid_res, 1);                    % 当前类别样本数
    mid_tiran = round(num_size * mid_size);         % 当前类别训练样本数（四舍五入）
    
    % 划分训练集输入和输出
    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 当前类别的训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 当前类别的训练集输出
    
    % 划分测试集输入和输出
    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 当前类别的测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 当前类别的测试集输出
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
[p_train, ps_input] = mapminmax(P_train, 0, 1);          % 对训练集输入进行归一化
p_test  = mapminmax('apply', P_test, ps_input);         % 使用训练集的归一化参数对测试集输入进行归一化
t_train = ind2vec(T_train);                             % 将训练集输出转换为向量（类别编码）
t_test  = ind2vec(T_test );                             % 将测试集输出转换为向量

%% 节点个数
inputnum  = size(p_train, 1);  % 输入层节点数
hiddennum = 6;                 % 隐藏层节点数（可调节）
outputnum = size(t_train, 1);  % 输出层节点数

%% 建立网络
net = newff(p_train, t_train, hiddennum);  % 创建前馈神经网络，隐藏层节点数为hiddennum

%% 设置训练参数
net.trainParam.epochs     = 1000;      % 最大训练次数
net.trainParam.goal       = 1e-6;      % 训练目标误差
net.trainParam.lr         = 0.01;      % 学习率
net.trainParam.showWindow = 0;         % 关闭训练过程窗口显示

%% 参数初始化
c1      = 4.494;       % 学习因子1（惯性权重）
c2      = 4.494;       % 学习因子2
maxgen  =   30;        % 最大迭代次数（粒子群更新次数）
sizepop =    5;        % 种群规模（粒子数量）
Vmax    =  1.0;        % 最大速度限制
Vmin    = -1.0;        % 最小速度限制
popmax  =  2.0;        % 粒子位置上限
popmin  = -2.0;        % 粒子位置下限

%% 节点总数
% 总参数数 = 输入层到隐藏层权重 + 隐藏层阈值 + 隐藏层到输出层权重 + 输出层阈值
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

%% 初始化种群和速度，并计算初始适应度
for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % 随机初始化粒子位置
    V(i, :) = rands(1, numsum);    % 随机初始化粒子速度
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);  % 计算粒子的适应度值
end

%% 个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);  % 找到全局最优适应度值及其索引
zbest = pop(bestindex, :);     % 全局最佳粒子位置
gbest = pop;                   % 初始化个体最佳位置（每个粒子的当前最佳）
fitnessgbest = fitness;        % 初始化个体最佳适应度值
BestFit = fitnesszbest;        % 初始化全局最佳适应度值记录

%% 迭代寻优（PSO主要过程）
for i = 1 : maxgen
    for j = 1 : sizepop
        
        %% 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        % 速度限制
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        %% 种群更新（位置更新）
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);  % 更新粒子位置（步长系数0.2）
        % 位置限制
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        %% 自适应变异
        pos = unidrnd(numsum);  % 随机选择一个位置进行变异
        if rand > 0.95
            pop(j, pos) = rands(1, 1);  % 以5%的概率进行随机变异
        end
        
        %% 适应度值计算
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);  % 重新计算适应度值
    
    end
    
    for j = 1 : sizepop
    
        %% 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);          % 更新个体最佳位置
            fitnessgbest(j) = fitness(j);     % 更新个体最佳适应度值
        end
    
        %% 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);                 % 更新全局最佳位置
            fitnesszbest = fitness(j);         % 更新全局最佳适应度值
        end
    
    end
    
    %% 记录全局最佳适应度值
    BestFit = [BestFit, fitnesszbest];    
end

%% 提取最优初始权值和阈值
% 根据全局最佳粒子位置zbest提取权重和阈值
w1 = zbest(1 : inputnum * hiddennum);  % 输入层到隐藏层的权重
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);  % 隐藏层的阈值
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum);  % 隐藏层到输出层的权重
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);  % 输出层的阈值

%% 网络赋值
% 将最优权重和阈值赋值给神经网络
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );    % 输入层到隐藏层的权重
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);    % 隐藏层到输出层的权重
net.b{1}     = reshape(B1, hiddennum, 1);           % 隐藏层的阈值
net.b{2}     = B2';                                 % 输出层的阈值

%% 打开训练窗口 
net.trainParam.showWindow = 1;        % 打开训练过程窗口显示

%% 网络训练
net = train(net, p_train, t_train);    % 使用训练集再次训练网络

%% 仿真预测
t_sim1 = sim(net, p_train);            % 使用训练集进行预测
t_sim2 = sim(net, p_test );            % 使用测试集进行预测

%% 数据反归一化
T_sim1 = vec2ind(t_sim1);               % 将训练集预测结果转换为类别索引
T_sim2 = vec2ind(t_sim2);               % 将测试集预测结果转换为类别索引

%% 数据排序
[T_train, index_1] = sort(T_train);     % 对训练集真实标签进行排序
[T_test , index_2] = sort(T_test );     % 对测试集真实标签进行排序

T_sim1 = T_sim1(index_1);                % 按排序索引调整训练集预测结果
T_sim2 = T_sim2(index_2);                % 按排序索引调整测试集预测结果

%% 性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;  % 计算训练集的分类准确率
error2 = sum((T_sim2 == T_test )) / N * 100 ;  % 计算测试集的分类准确率

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

%% 误差曲线迭代图
figure
plot(1: length(BestFit), BestFit, 'LineWidth', 1.5);
xlabel('粒子群迭代次数');
ylabel('适应度值');
xlim([1, length(BestFit)])
string = {'模型迭代误差变化'};
title(string)
grid on

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
```

---

### 代码说明

1. **`fun.m` 文件**：
   - 该函数用于计算每个粒子的适应度值，即分类错误率。它通过将粒子的位置信息（即神经网络的权重和阈值）赋值给神经网络，训练网络并进行仿真测试，最终计算预测结果与真实标签的匹配程度。

2. **`main.m` 文件**：
   - **数据处理**：读取数据集，随机打乱，划分训练集和测试集，并进行归一化处理。
   - **神经网络构建**：创建BP神经网络，并设置其参数。
   - **粒子群优化**：初始化粒子群，迭代更新粒子的速度和位置，通过自适应变异机制增强搜索能力，寻找最优的权重和阈值组合。
   - **模型训练与测试**：使用优化后的参数训练神经网络，并对训练集和测试集进行预测，计算准确率。
   - **结果展示**：绘制预测结果对比图、误差迭代曲线以及混淆矩阵，直观展示模型性能。

通过结合PSO和BP算法，该代码旨在提高神经网络分类的准确性和训练效率。
