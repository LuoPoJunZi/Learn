### PSO-BP时序预测详细介绍

#### 什么是PSO-BP时序预测？

**PSO-BP时序预测**（Particle Swarm Optimization - Backpropagation Time Series Prediction）是一种结合**粒子群优化算法（Particle Swarm Optimization, PSO）**和**反向传播神经网络（Backpropagation Neural Network, BP）**的时间序列预测方法。该方法利用PSO算法优化BP神经网络的权重和偏置，从而提高模型的预测准确性和收敛速度。

#### PSO-BP时序预测的组成部分

1. **数据预处理**：
   - **数据导入与整理**：读取时间序列数据，处理缺失值和异常值，确保数据质量。
   - **数据构造**：利用延迟步长（lag）将时间序列数据转换为监督学习问题的输入输出对，构建特征矩阵和目标向量。
   - **数据归一化**：对输入数据和目标变量进行归一化或标准化处理，以加快训练速度和提高模型稳定性。

2. **BP神经网络构建**：
   - **初始化网络**：定义BP神经网络的结构，包括输入层、隐藏层和输出层的节点数。
   - **设置训练参数**：设定训练参数，如最大训练次数、目标误差、学习率等。

3. **PSO算法优化**：
   - **参数初始化**：初始化PSO算法的参数，包括种群规模、最大迭代次数、粒子速度和位置的边界等。
   - **适应度函数设计**：定义适应度函数，通过计算BP网络在训练集上的预测误差来评估粒子的适应度。
   - **粒子更新**：根据PSO算法的更新规则，调整粒子的速度和位置，搜索最优解。
   - **个体与全局最优更新**：更新每个粒子的个体最佳位置和全局最佳位置，逐步逼近最优解。

4. **模型训练与预测**：
   - **网络权重与偏置赋值**：使用PSO算法找到的最优权重和偏置初始化BP神经网络。
   - **模型训练**：使用训练集数据进一步训练BP神经网络，优化模型参数。
   - **模型预测**：使用训练好的BP神经网络对训练集和测试集数据进行预测，得到预测结果。

5. **结果分析与可视化**：
   - **性能指标计算**：计算预测误差和其他性能指标（如RMSE、R²、MAE等），评估模型的预测准确性和泛化能力。
   - **绘制对比图**：绘制训练集和测试集的真实值与预测值对比图、误差曲线迭代图以及散点图，直观展示模型的预测效果和优化过程。

#### PSO-BP时序预测的优势

1. **全局搜索能力强**：
   - PSO算法具有优秀的全局搜索能力，能够有效避免BP神经网络训练过程中陷入局部最优解的问题。

2. **收敛速度快**：
   - PSO算法通过群体协作的方式加速搜索过程，提高了BP神经网络权重优化的收敛速度。

3. **适应性强**：
   - PSO-BP模型可以根据不同的数据集和预测任务灵活调整参数，适应性强，适用于多种时间序列预测场景。

4. **简单易实现**：
   - PSO算法结构简单，易于与BP神经网络结合，实现相对容易。

#### PSO-BP时序预测的应用

PSO-BP时序预测广泛应用于各类需要高精度时间序列预测的领域，包括但不限于：

1. **金融预测**：
   - **股市价格预测**：预测股票市场的未来价格走势，辅助投资决策。
   - **经济指标预测**：预测GDP、通胀率等宏观经济指标，为政策制定提供参考。

2. **能源与电力**：
   - **电力负荷预测**：预测未来电力需求，优化电网调度和资源分配。
   - **能源消耗预测**：预测能源消耗趋势，辅助能源管理和规划。

3. **工程与制造**：
   - **设备故障预测**：预测设备的潜在故障，进行预防性维护，减少停机时间。
   - **生产过程控制**：拟合和预测制造过程中关键参数，优化生产流程，确保产品质量。

4. **环境科学**：
   - **气象预测**：预测未来的气温、降水量等气象指标，辅助天气预报。
   - **污染物浓度预测**：预测空气或水体中的污染物浓度，进行环境监测和管理。

5. **医疗健康**：
   - **疾病风险预测**：预测个体患某种疾病的风险，辅助医疗决策和健康管理。
   - **医疗费用预测**：预测患者的医疗费用支出，优化医疗资源分配。

6. **市场营销**：
   - **销售预测**：预测产品的未来销售量，优化库存管理和市场策略。
   - **客户需求预测**：预测客户的购买行为和需求变化，制定精准的营销策略。

#### 如何使用PSO-BP时序预测

使用PSO-BP时序预测模型主要包括以下步骤：

1. **准备数据集**：
   - **数据收集与整理**：确保时间序列数据的完整性和准确性，处理缺失值和异常值。
   - **数据构造**：利用延迟步长（lag）将时间序列数据转换为监督学习问题的输入输出对，构建特征矩阵和目标向量。
   - **数据归一化**：对输入数据和目标变量进行归一化或标准化处理，以加快训练速度和提高模型稳定性。

2. **构建BP神经网络**：
   - **初始化网络**：定义BP神经网络的结构，包括输入层、隐藏层和输出层的节点数。
   - **设置训练参数**：设定训练参数，如最大训练次数、目标误差、学习率等。

3. **PSO算法优化**：
   - **参数初始化**：初始化PSO算法的参数，包括种群规模、最大迭代次数、粒子速度和位置的边界等。
   - **适应度函数设计**：定义适应度函数，通过计算BP网络在训练集上的预测误差来评估粒子的适应度。
   - **粒子更新**：根据PSO算法的更新规则，调整粒子的速度和位置，搜索最优解。
   - **个体与全局最优更新**：更新每个粒子的个体最佳位置和全局最佳位置，逐步逼近最优解。

4. **模型训练与预测**：
   - **网络权重与偏置赋值**：使用PSO算法找到的最优权重和偏置初始化BP神经网络。
   - **模型训练**：使用训练集数据进一步训练BP神经网络，优化模型参数。
   - **模型预测**：使用训练好的BP神经网络对训练集和测试集数据进行预测，得到预测结果。

5. **模型评估与优化**：
   - **计算性能指标**：计算RMSE、R²、MAE、MBE、MAPE等指标，全面评估模型的性能。
   - **优化模型参数**：根据性能指标调整BP神经网络的参数（如隐藏层神经元数量、学习率等），进一步优化模型性能。

6. **结果分析与可视化**：
   - **预测结果对比图**：绘制训练集和测试集的真实值与预测值对比图，直观展示模型的预测效果。
   - **误差曲线迭代图**：绘制PSO算法优化过程中适应度值的变化曲线，了解优化过程的收敛情况。
   - **散点图**：绘制真实值与预测值的散点图，评估模型的拟合能力和误差分布。
   - **误差分析**：通过计算并分析RMSE、R²、MAE、MBE、MAPE等指标，全面评估模型的性能和预测准确性。

#### PSO-BP时序预测的优势

1. **全局优化能力**：
   - PSO算法能够有效搜索全局最优解，避免BP神经网络训练过程中陷入局部最优的问题，提高模型的预测准确性。

2. **收敛速度快**：
   - PSO算法通过群体协作的方式加速搜索过程，提高了BP神经网络权重优化的收敛速度，缩短训练时间。

3. **简单易实现**：
   - PSO算法结构简单，易于与BP神经网络结合，实现相对容易，适用于各种时间序列预测任务。

4. **适应性强**：
   - PSO-BP模型可以根据不同的数据集和预测任务灵活调整参数，适应性强，适用于多种时间序列预测场景。

#### 如何使用PSO-BP时序预测

使用PSO-BP时序预测模型主要包括以下步骤：

1. **准备数据集**：
   - **数据收集与整理**：确保时间序列数据的完整性和准确性，处理缺失值和异常值。
   - **数据构造**：利用延迟步长（lag）将时间序列数据转换为监督学习问题的输入输出对，构建特征矩阵和目标向量。
   - **数据归一化**：对输入数据和目标变量进行归一化或标准化处理，以加快训练速度和提高模型稳定性。

2. **构建BP神经网络**：
   - **初始化网络**：定义BP神经网络的结构，包括输入层、隐藏层和输出层的节点数。
   - **设置训练参数**：设定训练参数，如最大训练次数、目标误差、学习率等。

3. **PSO算法优化**：
   - **参数初始化**：初始化PSO算法的参数，包括种群规模、最大迭代次数、粒子速度和位置的边界等。
   - **适应度函数设计**：定义适应度函数，通过计算BP网络在训练集上的预测误差来评估粒子的适应度。
   - **粒子更新**：根据PSO算法的更新规则，调整粒子的速度和位置，搜索最优解。
   - **个体与全局最优更新**：更新每个粒子的个体最佳位置和全局最佳位置，逐步逼近最优解。

4. **模型训练与预测**：
   - **网络权重与偏置赋值**：使用PSO算法找到的最优权重和偏置初始化BP神经网络。
   - **模型训练**：使用训练集数据进一步训练BP神经网络，优化模型参数。
   - **模型预测**：使用训练好的BP神经网络对训练集和测试集数据进行预测，得到预测结果。

5. **模型评估与优化**：
   - **计算性能指标**：计算RMSE、R²、MAE、MBE、MAPE等指标，全面评估模型的性能。
   - **优化模型参数**：根据性能指标调整BP神经网络的参数（如隐藏层神经元数量、学习率等），进一步优化模型性能。

6. **结果分析与可视化**：
   - **预测结果对比图**：绘制训练集和测试集的真实值与预测值对比图，直观展示模型的预测效果。
   - **误差曲线迭代图**：绘制PSO算法优化过程中适应度值的变化曲线，了解优化过程的收敛情况。
   - **散点图**：绘制真实值与预测值的散点图，评估模型的拟合能力和误差分布。
   - **误差分析**：通过计算并分析RMSE、R²、MAE、MBE、MAPE等指标，全面评估模型的性能和预测准确性。

---

### 代码简介

该MATLAB代码实现了基于**粒子群优化算法（PSO）**优化**反向传播神经网络（BP）**的时间序列预测算法，简称“PSO-BP时序预测”。主要流程如下：

1. **数据预处理**：
   - 导入时间序列数据，并构造监督学习的数据集。
   - 将数据集划分为训练集和测试集。
   - 对输入数据和目标变量进行归一化处理。

2. **BP神经网络构建与初始化**：
   - 定义BP神经网络的结构，包括输入层、隐藏层和输出层的节点数。
   - 设置BP神经网络的训练参数。

3. **PSO算法优化**：
   - 初始化PSO算法的参数，包括种群规模、最大迭代次数、粒子速度和位置的边界等。
   - 初始化种群和速度。
   - 运行PSO算法，通过迭代优化粒子的速度和位置，寻找最优的BP神经网络权重和偏置。

4. **模型训练与预测**：
   - 使用PSO算法找到的最优权重和偏置初始化BP神经网络。
   - 使用训练集数据训练BP神经网络，进一步优化模型参数。
   - 使用训练好的BP神经网络对训练集和测试集数据进行预测，得到预测结果。

5. **结果分析与可视化**：
   - 计算并显示相关回归性能指标（RMSE、R²、MAE、MBE、MAPE）。
   - 绘制训练集和测试集的真实值与预测值对比图、误差曲线迭代图以及散点图，直观展示模型的预测效果和优化过程。

以下是包含详细中文注释的PSO-BP时序预测MATLAB代码。

---

### MATLAB代码（添加详细中文注释）

#### fun.m 文件代码

```matlab
function error = fun(pop, hiddennum, net, p_train, t_train)
% FUN 适应度函数，用于评估粒子群优化算法中每个粒子的适应度
% 输入参数：
%   pop       - 当前粒子的编码（权重和偏置的向量）
%   hiddennum - 隐藏层节点数
%   net       - BP神经网络模型
%   p_train   - 训练集输入特征
%   t_train   - 训练集目标变量
% 输出参数：
%   error     - 当前粒子的适应度值（RMSE）

    %% 节点个数
    inputnum  = size(p_train, 1);   % 输入层节点数
    outputnum = size(t_train, 1);   % 输出层节点数
    
    %% 提取权值和阈值
    w1 = pop(1 : inputnum * hiddennum);  % 提取输入层到隐藏层的权重向量
    B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum); % 提取隐藏层的偏置向量
    w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum); % 提取隐藏层到输出层的权重向量
    B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum); % 提取输出层的偏置向量
     
    %% 网络赋值
    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );  % 将权重向量w1重塑为隐藏层权重矩阵
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum); % 将权重向量w2重塑为输出层权重矩阵
    net.b{1}     = reshape(B1, hiddennum, 1);        % 将偏置向量B1重塑为隐藏层偏置
    net.b{2}     = B2';                               % 将偏置向量B2转置为输出层偏置
    
    %% 网络训练
    net = train(net, p_train, t_train); % 使用训练集数据训练BP神经网络，更新网络权重和偏置
    
    %% 仿真测试
    t_sim1 = sim(net, p_train);  % 使用训练集数据进行仿真预测，得到训练集预测结果
    
    %% 适应度值
    error = sqrt(sum((t_sim1 - t_train) .^ 2) ./ length(t_sim1)); % 计算均方根误差（RMSE）作为适应度值
end
```

#### main.m 文件代码

```matlab
%% 清空环境变量
warning off             % 关闭所有警告信息，避免运行过程中显示不必要的警告
close all               % 关闭所有打开的图形窗口，确保绘图环境的干净
clear                   % 清除工作区中的所有变量，确保没有残留变量影响结果
clc                     % 清空命令行窗口，提升可读性

%% 导入数据（时间序列的单列数据）
result = xlsread('数据集.xlsx');  % 从Excel文件中读取时间序列数据，假设数据为单列

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

%% 节点个数
inputnum  = size(p_train, 1);  % 输入层节点数
hiddennum = 5;                 % 隐藏层节点数
outputnum = size(t_train, 1);  % 输出层节点数

%% 建立网络
net = newff(p_train, t_train, hiddennum); % 创建一个前馈神经网络，使用newff函数
% 注意：newff函数在较新版本的MATLAB中可能已弃用，建议使用feedforwardnet替代

%% 设置训练参数
net.trainParam.epochs     = 1000;      % 设置BP网络的最大训练次数为1000
net.trainParam.goal       = 1e-6;      % 设置训练的误差目标为1e-6，达到此误差后停止训练
net.trainParam.lr         = 0.01;      % 设置BP网络的学习率为0.01，控制权重更新的步长大小
net.trainParam.showWindow = 0;         % 关闭训练窗口，避免训练过程中的图形干扰

%% 参数初始化
c1      = 4.494;       % PSO算法的学习因子1
c2      = 4.494;       % PSO算法的学习因子2
maxgen  =   30;        % PSO算法的最大迭代次数
sizepop =    5;        % PSO算法的种群规模（每代的粒子数量）
Vmax    =  1.0;        % 粒子的最大速度
Vmin    = -1.0;        % 粒子的最小速度
popmax  =  2.0;        % 粒子位置的最大边界
popmin  = -2.0;        % 粒子位置的最小边界

%% 节点总数
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;
% 总优化参数个数，包括输入层到隐藏层的权重、隐藏层的偏置、隐藏层到输出层的权重和输出层的偏置

%% 初始化种群和速度
for i = 1:sizepop
    pop(i, :) = rands(1, numsum);  % 初始化种群，每个粒子的位置随机生成
    V(i, :) = rands(1, numsum);    % 初始化速度，每个粒子的速度随机生成
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train); % 计算每个粒子的适应度
end

%% 个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);  % 找到适应度最优的粒子
zbest = pop(bestindex, :);     % 全局最佳位置，即适应度最优的粒子的位置
gbest = pop;                   % 个体最佳位置，初始化为当前种群的位置
fitnessgbest = fitness;        % 个体最佳适应度值，初始化为当前种群的适应度
BestFit = fitnesszbest;        % 全局最佳适应度值，初始化为当前最优适应度

%% 迭代寻优
for i = 1:maxgen
    for j = 1:sizepop
        
        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;     % 限制速度不超过Vmax
        V(j, (V(j, :) < Vmin)) = Vmin;     % 限制速度不低于Vmin
        
        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :); % 更新粒子的位置
        pop(j, (pop(j, :) > popmax)) = popmax; % 限制位置不超过popmax
        pop(j, (pop(j, :) < popmin)) = popmin; % 限制位置不低于popmin
        
        % 自适应变异
        pos = unidrnd(numsum);                 % 随机选择一个位置进行变异
        if rand > 0.95
            pop(j, pos) = rands(1, 1);         % 以5%的概率对选定位置进行随机变异
        end
        
        % 适应度值
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train); % 重新计算适应度
    end
    
    for j = 1:sizepop

        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);             % 更新个体最佳位置
            fitnessgbest(j) = fitness(j);        % 更新个体最佳适应度值
        end

        % 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);                    % 更新全局最佳位置
            fitnesszbest = fitness(j);            % 更新全局最佳适应度值
        end

    end

    BestFit = [BestFit, fitnesszbest];    % 记录每代的全局最佳适应度值
end

%% 提取最优初始权值和阈值
w1 = zbest(1 : inputnum * hiddennum);  % 提取全局最佳粒子的位置对应的输入层到隐藏层权重
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum); % 提取隐藏层的偏置
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum); % 提取隐藏层到输出层的权重
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum); % 提取输出层的偏置

%% 最优值赋值
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);    % 设置BP网络的输入层到隐藏层的权重矩阵
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);   % 设置BP网络的隐藏层到输出层的权重矩阵
net.b{1}     = reshape(B1, hiddennum, 1);          % 设置BP网络的隐藏层偏置
net.b{2}     = B2';                                 % 设置BP网络的输出层偏置

%% 打开训练窗口 
net.trainParam.showWindow = 1;        % 打开训练窗口，显示训练过程

%% 网络训练
net = train(net, p_train, t_train);    % 使用训练集数据训练BP神经网络，优化网络权重和偏置

%% 仿真预测
t_sim1 = sim(net, p_train);             % 使用训练集数据进行仿真预测，得到训练集预测结果
t_sim2 = sim(net, p_test );             % 使用测试集数据进行仿真预测，得到测试集预测结果

%% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);  % 将训练集预测结果反归一化，恢复到原始尺度
T_sim2 = mapminmax('reverse', t_sim2, ps_output);  % 将测试集预测结果反归一化，恢复到原始尺度

%% 均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);  % 计算训练集的均方根误差（RMSE）
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);  % 计算测试集的均方根误差（RMSE）

%% 查看网络结构
analyzeNetwork(net) % 可视化和分析BP神经网络的结构和参数

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

%% 误差曲线迭代图
figure
plot(1:length(BestFit), BestFit, 'LineWidth', 1.5);          % 绘制适应度值随迭代次数的变化曲线
xlabel('粒子群迭代次数');                                      % 设置X轴标签为“粒子群迭代次数”
ylabel('适应度值');                                            % 设置Y轴标签为“适应度值”
xlim([1, length(BestFit)])                                     % 设置X轴显示范围为[1, 迭代次数]
string = {'模型迭代误差变化'};                                 % 设置图形标题
title(string)                                                    % 添加图形标题
grid on                                                          % 显示网格，提升图形的可读性

%% 相关指标计算
% 决定系数（R²）
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;  % 计算训练集的决定系数R²
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;    % 计算测试集的决定系数R²

disp(['训练集数据的R2为：', num2str(R1)])  % 显示训练集的R²
disp(['测试集数据的R2为：', num2str(R2)])  % 显示测试集的R²

% 平均绝对误差（MAE）
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;  % 计算训练集的平均绝对误差MAE
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;  % 计算测试集的平均绝对误差MAE

disp(['训练集数据的MAE为：', num2str(mae1)])  % 显示训练集的MAE
disp(['测试集数据的MAE为：', num2str(mae2)])  % 显示测试集的MAE

% 平均偏差误差（MBE）
mbe1 = sum(T_sim1 - T_train) ./ M ;  % 计算训练集的平均偏差误差MBE
mbe2 = sum(T_sim2 - T_test ) ./ N ;  % 计算测试集的平均偏差误差MBE

disp(['训练集数据的MBE为：', num2str(mbe1)])  % 显示训练集的MBE
disp(['测试集数据的MBE为：', num2str(mbe2)])  % 显示测试集的MBE

% 平均绝对百分比误差（MAPE）
mape1 = sum(abs((T_sim1 - T_train)./T_train)) ./ M ;  % 计算训练集的平均绝对百分比误差MAPE
mape2 = sum(abs((T_sim2 - T_test )./T_test )) ./ N ;  % 计算测试集的平均绝对百分比误差MAPE

disp(['训练集数据的MAPE为：', num2str(mape1)])  % 显示训练集的MAPE
disp(['测试集数据的MAPE为：', num2str(mape2)])  % 显示测试集的MAPE

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
```

---

### 代码说明

#### 1. 清空环境变量

```matlab
warning off             % 关闭所有警告信息，避免运行过程中显示不必要的警告
close all               % 关闭所有打开的图形窗口，确保绘图环境的干净
clear                   % 清除工作区中的所有变量，确保没有残留变量影响结果
clc                     % 清空命令行窗口，提升可读性
```

- **warning off**：关闭MATLAB中的所有警告信息，避免在代码运行过程中显示不必要的警告。
- **close all**：关闭所有打开的图形窗口，避免之前的图形干扰当前的绘图。
- **clear**：清除工作区中的所有变量，确保代码运行环境的干净。
- **clc**：清空命令行窗口，提升可读性。

#### 2. 导入数据

```matlab
result = xlsread('数据集.xlsx');  % 从Excel文件中读取时间序列数据，假设数据为单列
```

- **xlsread**：从指定的Excel文件`数据集.xlsx`中读取时间序列数据。
- **result**：存储读取的时间序列数据，假设数据为单列，表示时间序列的连续值。

#### 3. 数据分析

```matlab
num_samples = length(result);  % 计算时间序列数据的样本数量（数据点数）
kim = 15;                      % 设定延时步长（lag），即使用15个历史数据点作为输入特征
zim =  1;                      % 设定预测步长（forecast step），即预测当前点之后的1个时间点
```

- **num_samples**：计算时间序列数据的样本数量，即数据点的总数。
- **kim**：设定延时步长（lag），即每次使用15个连续的历史数据点作为输入特征，用于预测未来的值。
- **zim**：设定预测步长（forecast step），即预测当前点之后的1个时间点的值。

#### 4. 构造数据集

```matlab
for i = 1:num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i:i + kim - 1), 1, kim), result(i + kim + zim - 1)];
end
```

- **循环构造数据集**：
  - 遍历时间序列数据，从第1个数据点到第`num_samples - kim - zim + 1`个数据点。
  - **reshape(result(i:i + kim - 1), 1, kim)**：将连续的`kim`个历史数据点转换为1行`kim`列的向量，作为输入特征。
  - **result(i + kim + zim - 1)**：获取当前输入特征对应的目标变量，即第`kim + zim`个时间点的值。
  - **res(i, :)**：将输入特征和目标变量组合成一行，存储在结果矩阵`res`中。

#### 5. 数据集分析

```matlab
outdim = 1;                                  % 设定数据集的最后一列为输出（目标变量）
num_size = 0.7;                              % 设定训练集占数据集的比例（70%训练集，30%测试集）
num_train_s = round(num_size * num_samples); % 计算训练集样本个数，通过四舍五入确定
f_ = size(res, 2) - outdim;                  % 计算输入特征的维度，即总列数减去输出维度
```

- **outdim**：设定数据集的最后一列为输出（目标变量）。
- **num_size**：设定训练集占数据集的比例为70%，剩余30%作为测试集。
- **num_train_s**：计算训练集的样本数量，通过`round`函数对训练集比例与总样本数的乘积进行四舍五入。
- **f_**：计算输入特征的维度，即数据集的总列数减去输出维度。

#### 6. 划分训练集和测试集

```matlab
P_train = res(1:num_train_s, 1:f_)';         % 训练集输入特征，转置使每列为一个样本 (f_ × M)
T_train = res(1:num_train_s, f_ + 1:end)';   % 训练集输出目标变量，转置使每列为一个样本 (outdim × M)
M = size(P_train, 2);                        % 获取训练集的样本数量

P_test = res(num_train_s + 1:end, 1:f_)';    % 测试集输入特征，转置使每列为一个样本 (f_ × N)
T_test = res(num_train_s + 1:end, f_ + 1:end)';% 测试集输出目标变量，转置使每列为一个样本 (outdim × N)
N = size(P_test, 2);                         % 获取测试集的样本数量
```

- **P_train**：提取前`num_train_s`个样本的输入特征，并进行转置，使每列为一个样本。
- **T_train**：提取前`num_train_s`个样本的输出（目标变量），并进行转置，使每列为一个样本。
- **M**：获取训练集的样本数量。
- **P_test**：提取剩余样本的输入特征，并进行转置，使每列为一个样本。
- **T_test**：提取剩余样本的输出（目标变量），并进行转置，使每列为一个样本。
- **N**：获取测试集的样本数量。

#### 7. 数据归一化

```matlab
[p_train, ps_input] = mapminmax(P_train, 0, 1);         % 对训练集输入特征进行归一化，范围[0,1]
p_test = mapminmax('apply', P_test, ps_input);           % 使用训练集的归一化参数对测试集输入特征进行归一化

[t_train, ps_output] = mapminmax(T_train, 0, 1);         % 对训练集输出目标变量进行归一化，范围[0,1]
t_test = mapminmax('apply', T_test, ps_output);           % 使用训练集的归一化参数对测试集输出目标变量进行归一化
```

- **mapminmax**：使用`mapminmax`函数将数据缩放到指定的范围内（这里为[0,1]）。
- **p_train**：归一化后的训练集输入特征数据。
- **ps_input**：保存输入特征的归一化参数，以便对测试集数据进行相同的归一化处理。
- **p_test**：使用训练集的归一化参数对测试集输入特征数据进行归一化，确保训练集和测试集的数据尺度一致。
- **t_train**：归一化后的训练集输出目标变量数据。
- **ps_output**：保存输出目标变量的归一化参数，以便对测试集数据进行相同的归一化处理。
- **t_test**：使用训练集的归一化参数对测试集输出目标变量数据进行归一化。

#### 8. 节点个数

```matlab
inputnum  = size(p_train, 1);  % 输入层节点数
hiddennum = 5;                 % 隐藏层节点数
outputnum = size(t_train, 1);  % 输出层节点数
```

- **inputnum**：输入层节点数，等于输入特征的维度。
- **hiddennum**：隐藏层节点数，设定为5，根据数据复杂度和模型需求可以调整。
- **outputnum**：输出层节点数，等于目标变量的维度。

#### 9. 建立网络

```matlab
net = newff(p_train, t_train, hiddennum); % 创建一个前馈神经网络，使用newff函数
% 注意：newff函数在较新版本的MATLAB中可能已弃用，建议使用feedforwardnet替代
```

- **newff**：创建一个前馈神经网络，输入层节点数与输入特征维度相同，输出层节点数与目标变量维度相同，隐藏层节点数为`hiddennum`。
- **注意**：在较新版本的MATLAB中，`newff`函数可能已被弃用，建议使用`feedforwardnet`函数替代。

#### 10. 设置训练参数

```matlab
net.trainParam.epochs     = 1000;      % 设置BP网络的最大训练次数为1000
net.trainParam.goal       = 1e-6;      % 设置训练的误差目标为1e-6，达到此误差后停止训练
net.trainParam.lr         = 0.01;      % 设置BP网络的学习率为0.01，控制权重更新的步长大小
net.trainParam.showWindow = 0;         % 关闭训练窗口，避免训练过程中的图形干扰
```

- **net.trainParam.epochs**：设置BP网络的最大训练次数为1000。
- **net.trainParam.goal**：设置训练的误差目标为1e-6，达到此误差后停止训练。
- **net.trainParam.lr**：设置BP网络的学习率为0.01，控制权重更新的步长大小。
- **net.trainParam.showWindow**：关闭训练窗口，避免训练过程中的图形干扰。

#### 11. 参数初始化

```matlab
c1      = 4.494;       % PSO算法的学习因子1
c2      = 4.494;       % PSO算法的学习因子2
maxgen  =   30;        % PSO算法的最大迭代次数
sizepop =    5;        % PSO算法的种群规模（每代的粒子数量）
Vmax    =  1.0;        % 粒子的最大速度
Vmin    = -1.0;        % 粒子的最小速度
popmax  =  2.0;        % 粒子位置的最大边界
popmin  = -2.0;        % 粒子位置的最小边界
```

- **c1, c2**：PSO算法的学习因子，控制粒子向个体最佳和全局最佳位置的移动速度。
- **maxgen**：PSO算法的最大迭代次数，设定为30。
- **sizepop**：PSO算法的种群规模（每代的粒子数量），设定为5。
- **Vmax, Vmin**：粒子的最大速度和最小速度，控制粒子的移动速度范围。
- **popmax, popmin**：粒子位置的最大边界和最小边界，限制粒子搜索空间。

#### 12. 节点总数

```matlab
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;
% 总优化参数个数，包括输入层到隐藏层的权重、隐藏层的偏置、隐藏层到输出层的权重和输出层的偏置
```

- **numsum**：计算优化参数的总个数，包括输入层到隐藏层的权重、隐藏层的偏置、隐藏层到输出层的权重和输出层的偏置。

#### 13. 初始化种群和速度

```matlab
for i = 1:sizepop
    pop(i, :) = rands(1, numsum);  % 初始化种群，每个粒子的位置随机生成
    V(i, :) = rands(1, numsum);    % 初始化速度，每个粒子的速度随机生成
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train); % 计算每个粒子的适应度
end
```

- **循环初始化种群**：
  - **pop(i, :)**：初始化种群的位置，每个粒子的位置由随机生成的`numsum`个数值组成。
  - **V(i, :)**：初始化粒子的速度，每个粒子的速度由随机生成的`numsum`个数值组成。
  - **fitness(i)**：计算每个粒子的适应度，调用`fun`函数，通过BP神经网络在训练集上的预测误差计算适应度值。

#### 14. 个体极值和群体极值

```matlab
[fitnesszbest, bestindex] = min(fitness);  % 找到适应度最优的粒子
zbest = pop(bestindex, :);     % 全局最佳位置，即适应度最优的粒子的位置
gbest = pop;                   % 个体最佳位置，初始化为当前种群的位置
fitnessgbest = fitness;        % 个体最佳适应度值，初始化为当前种群的适应度
BestFit = fitnesszbest;        % 全局最佳适应度值，初始化为当前最优适应度
```

- **[fitnesszbest, bestindex] = min(fitness)**：找到适应度最优的粒子，获取最小适应度值和对应的粒子索引。
- **zbest**：全局最佳位置，即适应度最优的粒子的位置。
- **gbest**：个体最佳位置，初始化为当前种群的位置。
- **fitnessgbest**：个体最佳适应度值，初始化为当前种群的适应度。
- **BestFit**：记录全局最佳适应度值，初始化为当前最优适应度。

#### 15. 迭代寻优

```matlab
for i = 1:maxgen
    for j = 1:sizepop
        
        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;     % 限制速度不超过Vmax
        V(j, (V(j, :) < Vmin)) = Vmin;     % 限制速度不低于Vmin
        
        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :); % 更新粒子的位置
        pop(j, (pop(j, :) > popmax)) = popmax; % 限制位置不超过popmax
        pop(j, (pop(j, :) < popmin)) = popmin; % 限制位置不低于popmin
        
        % 自适应变异
        pos = unidrnd(numsum);                 % 随机选择一个位置进行变异
        if rand > 0.95
            pop(j, pos) = rands(1, 1);         % 以5%的概率对选定位置进行随机变异
        end
        
        % 适应度值
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train); % 重新计算适应度
    end
    
    for j = 1:sizepop

        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);             % 更新个体最佳位置
            fitnessgbest(j) = fitness(j);        % 更新个体最佳适应度值
        end

        % 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);                    % 更新全局最佳位置
            fitnesszbest = fitness(j);            % 更新全局最佳适应度值
        end

    end

    BestFit = [BestFit, fitnesszbest];    % 记录每代的全局最佳适应度值
end
```

- **外层循环**：迭代寻优，执行`maxgen`次迭代。
- **内层循环**：遍历每个粒子，更新速度和位置。
  - **速度更新**：
    - 根据PSO算法的公式更新粒子的速度，结合个体最佳位置和全局最佳位置。
    - 限制速度不超过`Vmax`和不低于`Vmin`。
  - **种群更新**：
    - 根据更新后的速度调整粒子的位置。
    - 限制位置不超过`popmax`和不低于`popmin`。
  - **自适应变异**：
    - 以5%的概率对粒子的位置进行随机变异，增加搜索的多样性。
  - **适应度值**：
    - 重新计算粒子的适应度值。
- **个体最优和全局最优更新**：
  - **个体最优更新**：如果当前粒子的适应度优于其个体最佳适应度，则更新个体最佳位置和适应度值。
  - **全局最优更新**：如果当前粒子的适应度优于全局最佳适应度，则更新全局最佳位置和适应度值。
- **记录全局最佳适应度值**：将每代的全局最佳适应度值记录在`BestFit`中，用于后续绘图。

#### 16. 提取最优初始权值和阈值

```matlab
w1 = zbest(1 : inputnum * hiddennum);  % 提取全局最佳粒子的位置对应的输入层到隐藏层权重
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum); % 提取隐藏层的偏置
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum); % 提取隐藏层到输出层的权重
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum); % 提取输出层的偏置
```

- **w1**：提取全局最佳粒子的位置对应的输入层到隐藏层的权重向量。
- **B1**：提取隐藏层的偏置向量。
- **w2**：提取隐藏层到输出层的权重向量。
- **B2**：提取输出层的偏置向量。

#### 17. 最优值赋值

```matlab
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);    % 设置BP网络的输入层到隐藏层的权重矩阵
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);   % 设置BP网络的隐藏层到输出层的权重矩阵
net.b{1}     = reshape(B1, hiddennum, 1);          % 设置BP网络的隐藏层偏置
net.b{2}     = B2';                                 % 设置BP网络的输出层偏置
```

- **net.Iw{1, 1}**：将权重向量`w1`重塑为隐藏层权重矩阵，并赋值给BP网络的输入层权重。
- **net.Lw{2, 1}**：将权重向量`w2`重塑为输出层权重矩阵，并赋值给BP网络的隐藏层到输出层的权重。
- **net.b{1}**：将偏置向量`B1`重塑为隐藏层偏置，并赋值给BP网络的隐藏层偏置。
- **net.b{2}**：将偏置向量`B2`转置为输出层偏置，并赋值给BP网络的输出层偏置。

#### 18. 打开训练窗口

```matlab
net.trainParam.showWindow = 1;        % 打开训练窗口，显示训练过程
```

- **net.trainParam.showWindow**：设置为1，打开训练窗口，显示BP网络的训练过程。

#### 19. 网络训练

```matlab
net = train(net, p_train, t_train);    % 使用训练集数据训练BP神经网络，优化网络权重和偏置
```

- **train**：使用训练集数据`p_train`和`t_train`训练BP神经网络，调整网络权重和偏置以最小化预测误差。

#### 20. 仿真预测

```matlab
t_sim1 = sim(net, p_train);             % 使用训练集数据进行仿真预测，得到训练集预测结果
t_sim2 = sim(net, p_test );             % 使用测试集数据进行仿真预测，得到测试集预测结果
```

- **sim**：使用训练好的BP神经网络对输入数据进行仿真预测。
- **t_sim1**：训练集的预测结果。
- **t_sim2**：测试集的预测结果。

#### 21. 数据反归一化

```matlab
T_sim1 = mapminmax('reverse', t_sim1, ps_output);  % 将训练集预测结果反归一化，恢复到原始尺度
T_sim2 = mapminmax('reverse', t_sim2, ps_output);  % 将测试集预测结果反归一化，恢复到原始尺度
```

- **mapminmax('reverse', ...)**：使用`mapminmax`函数将预测结果反归一化，恢复到原始数据的尺度。
- **T_sim1**：训练集预测结果，恢复到原始尺度。
- **T_sim2**：测试集预测结果，恢复到原始尺度。

#### 22. 均方根误差（RMSE）

```matlab
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);  % 计算训练集的均方根误差（RMSE）
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);  % 计算测试集的均方根误差（RMSE）
```

- **RMSE**：均方根误差，衡量模型预测值与真实值之间的平均差异。
- **error1**：
  - 训练集的RMSE，计算公式为：
    \[
    RMSE = \sqrt{\frac{1}{M} \sum_{i=1}^{M} (T_{\text{sim1}} - T_{\text{train}})^2}
    \]
- **error2**：
  - 测试集的RMSE，计算公式为：
    \[
    RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (T_{\text{sim2}} - T_{\text{test}})^2}
    \]

#### 23. 查看网络结构

```matlab
analyzeNetwork(net) % 可视化和分析BP神经网络的结构和参数
```

- **analyzeNetwork**：可视化和分析训练好的BP神经网络结构和参数，包括层次结构、连接权重等，帮助用户理解和优化网络。

#### 24. 绘图

##### 绘制训练集预测结果对比图

```matlab
figure
plot(1:M, T_train, 'r-', 1:M, T_sim1, 'b-', 'LineWidth', 1) % 绘制训练集真实值与预测值的对比曲线，红色实线为真实值，蓝色实线为预测值
legend('真实值', '预测值')                                        % 添加图例，区分真实值和预测值
xlabel('预测样本')                                                % 设置X轴标签为“预测样本”
ylabel('预测结果')                                                % 设置Y轴标签为“预测结果”
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};      % 创建标题字符串，包括RMSE值
title(string)                                                    % 添加图形标题
xlim([1, M])                                                     % 设置X轴显示范围为[1, M]
grid                                                             % 显示网格，提升图形的可读性
```

- **figure**：创建新的图形窗口。
- **plot(1:M, T_train, 'r-', 1:M, T_sim1, 'b-', 'LineWidth', 1)**：
  - 绘制训练集真实值`T_train`与预测值`t_sim1`的对比曲线，红色实线表示真实值，蓝色实线表示预测值。
- **legend('真实值', '预测值')**：
  - 添加图例，区分真实值和预测值。
- **xlabel('预测样本')** 和 **ylabel('预测结果')**：
  - 设置X轴和Y轴的标签为“预测样本”和“预测结果”。
- **string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};**：
  - 创建标题字符串，包括RMSE值。
- **title(string)**：
  - 添加图形标题。
- **xlim([1, M])**：
  - 设置X轴的显示范围为[1, M]，其中M为训练集样本数。
- **grid**：
  - 显示网格，提升图形的可读性。

##### 绘制测试集预测结果对比图

```matlab
figure
plot(1:N, T_test, 'r-', 1:N, T_sim2, 'b-', 'LineWidth', 1)   % 绘制测试集真实值与预测值的对比曲线，红色实线为真实值，蓝色实线为预测值
legend('真实值', '预测值')                                        % 添加图例，区分真实值和预测值
xlabel('预测样本')                                                % 设置X轴标签为“预测样本”
ylabel('预测结果')                                                % 设置Y轴标签为“预测结果”
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};       % 创建标题字符串，包括RMSE值
title(string)                                                    % 添加图形标题
xlim([1, N])                                                     % 设置X轴显示范围为[1, N]
grid                                                             % 显示网格，提升图形的可读性
```

- **figure**：创建新的图形窗口。
- **plot(1:N, T_test, 'r-', 1:N, T_sim2, 'b-', 'LineWidth', 1)**：
  - 绘制测试集真实值`T_test`与预测值`t_sim2`的对比曲线，红色实线表示真实值，蓝色实线表示预测值。
- **legend('真实值', '预测值')**：
  - 添加图例，区分真实值和预测值。
- **xlabel('预测样本')** 和 **ylabel('预测结果')**：
  - 设置X轴和Y轴的标签为“预测样本”和“预测结果”。
- **string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};**：
  - 创建标题字符串，包括RMSE值。
- **title(string)**：
  - 添加图形标题。
- **xlim([1, N])**：
  - 设置X轴的显示范围为[1, N]，其中N为测试集样本数。
- **grid**：
  - 显示网格，提升图形的可读性。

#### 25. 误差曲线迭代图

```matlab
figure
plot(1:length(BestFit), BestFit, 'LineWidth', 1.5);          % 绘制适应度值随迭代次数的变化曲线
xlabel('粒子群迭代次数');                                      % 设置X轴标签为“粒子群迭代次数”
ylabel('适应度值');                                            % 设置Y轴标签为“适应度值”
xlim([1, length(BestFit)])                                     % 设置X轴显示范围为[1, 迭代次数]
string = {'模型迭代误差变化'};                                 % 设置图形标题
title(string)                                                    % 添加图形标题
grid on                                                          % 显示网格，提升图形的可读性
```

- **figure**：创建新的图形窗口。
- **plot(1:length(BestFit), BestFit, 'LineWidth', 1.5)**：
  - 绘制适应度值随迭代次数的变化曲线，横轴为迭代次数，纵轴为适应度值（RMSE）。
- **xlabel('粒子群迭代次数')** 和 **ylabel('适应度值')**：
  - 设置X轴和Y轴的标签为“粒子群迭代次数”和“适应度值”。
- **xlim([1, length(BestFit)])**：
  - 设置X轴的显示范围为[1, 迭代次数]。
- **string = {'模型迭代误差变化'};**：
  - 设置图形标题为“模型迭代误差变化”。
- **title(string)**：
  - 添加图形标题。
- **grid on**：
  - 显示网格，提升图形的可读性。

#### 26. 相关指标计算

```matlab
% 决定系数（R²）
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;  % 计算训练集的决定系数R²
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;    % 计算测试集的决定系数R²

disp(['训练集数据的R2为：', num2str(R1)])  % 显示训练集的R²
disp(['测试集数据的R2为：', num2str(R2)])  % 显示测试集的R²

% 平均绝对误差（MAE）
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;  % 计算训练集的平均绝对误差MAE
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;  % 计算测试集的平均绝对误差MAE

disp(['训练集数据的MAE为：', num2str(mae1)])  % 显示训练集的MAE
disp(['测试集数据的MAE为：', num2str(mae2)])  % 显示测试集的MAE

% 平均偏差误差（MBE）
mbe1 = sum(T_sim1 - T_train) ./ M ;  % 计算训练集的平均偏差误差MBE
mbe2 = sum(T_sim2 - T_test ) ./ N ;  % 计算测试集的平均偏差误差MBE

disp(['训练集数据的MBE为：', num2str(mbe1)])  % 显示训练集的MBE
disp(['测试集数据的MBE为：', num2str(mbe2)])  % 显示测试集的MBE

% 平均绝对百分比误差（MAPE）
mape1 = sum(abs((T_sim1 - T_train)./T_train)) ./ M ;  % 计算训练集的平均绝对百分比误差MAPE
mape2 = sum(abs((T_sim2 - T_test )./T_test )) ./ N ;  % 计算测试集的平均绝对百分比误差MAPE

disp(['训练集数据的MAPE为：', num2str(mape1)])  % 显示训练集的MAPE
disp(['测试集数据的MAPE为：', num2str(mape2)])  % 显示测试集的MAPE

% 均方根误差（RMSE）
disp(['训练集数据的RMSE为：', num2str(error1)])  % 显示训练集的RMSE
disp(['测试集数据的RMSE为：', num2str(error2)])  % 显示测试集的RMSE
```

- **决定系数（R²）**：
  - **R1**：训练集的决定系数R²，衡量模型对训练数据的拟合程度。值越接近1，表示模型对数据的解释能力越强。
  - **R2**：测试集的决定系数R²，衡量模型对测试数据的泛化能力。值越接近1，表示模型在未见数据上的表现越好。
  - **disp(['训练集数据的R2为：', num2str(R1)])**：
    - 显示训练集的R²值。
  - **disp(['测试集数据的R2为：', num2str(R2)])**：
    - 显示测试集的R²值。

- **平均绝对误差（MAE）**：
  - **mae1**：训练集的平均绝对误差MAE，表示预测值与真实值之间的平均绝对差异。值越小，表示模型性能越好。
  - **mae2**：测试集的平均绝对误差MAE，表示预测值与真实值之间的平均绝对差异。值越小，表示模型性能越好。
  - **disp(['训练集数据的MAE为：', num2str(mae1)])**：
    - 显示训练集的MAE值。
  - **disp(['测试集数据的MAE为：', num2str(mae2)])**：
    - 显示测试集的MAE值。

- **平均偏差误差（MBE）**：
  - **mbe1**：训练集的平均偏差误差MBE，衡量模型是否存在系统性偏差。正值表示模型倾向于高估，负值表示模型倾向于低估。
  - **mbe2**：测试集的平均偏差误差MBE，衡量模型是否存在系统性偏差。正值表示模型倾向于高估，负值表示模型倾向于低估。
  - **disp(['训练集数据的MBE为：', num2str(mbe1)])**：
    - 显示训练集的MBE值。
  - **disp(['测试集数据的MBE为：', num2str(mbe2)])**：
    - 显示测试集的MBE值。

- **平均绝对百分比误差（MAPE）**：
  - **mape1**：训练集的平均绝对百分比误差MAPE，表示预测值与真实值之间的平均绝对百分比差异。适用于评估相对误差。
  - **mape2**：测试集的平均绝对百分比误差MAPE，表示预测值与真实值之间的平均绝对百分比差异。适用于评估相对误差。
  - **disp(['训练集数据的MAPE为：', num2str(mape1)])**：
    - 显示训练集的MAPE值。
  - **disp(['测试集数据的MAPE为：', num2str(mape2)])**：
    - 显示测试集的MAPE值。

- **均方根误差（RMSE）**：
  - **error1**：训练集的RMSE，显示训练集的均方根误差。
  - **error2**：测试集的RMSE，显示测试集的均方根误差。
  - **disp(['训练集数据的RMSE为：', num2str(error1)])**：
    - 显示训练集的RMSE值。
  - **disp(['测试集数据的RMSE为：', num2str(error2)])**：
    - 显示测试集的RMSE值。

#### 27. 绘制散点图

##### 绘制训练集散点图

```matlab
figure
scatter(T_train, T_sim1, sz, c)              % 绘制训练集真实值与预测值的散点图，蓝色散点表示预测结果
hold on                                       % 保持当前图形，允许在同一图形上绘制多条曲线
plot(xlim, ylim, '--k')                      % 绘制理想预测线（真实值等于预测值的对角线），使用黑色虚线表示
xlabel('训练集真实值');                        % 设置X轴标签为“训练集真实值”
ylabel('训练集预测值');                        % 设置Y轴标签为“训练集预测值”
xlim([min(T_train) max(T_train)])             % 设置X轴的显示范围为[最小真实值, 最大真实值]
ylim([min(T_sim1) max(T_sim1)])               % 设置Y轴的显示范围为[最小预测值, 最大预测值]
title('训练集预测值 vs. 训练集真实值')            % 设置图形的标题为“训练集预测值 vs. 训练集真实值”
```

- **figure**：创建新的图形窗口。
- **scatter(T_train, T_sim1, sz, c)**：
  - 使用`scatter`函数绘制训练集真实值`T_train`与预测值`t_sim1`的散点图，蓝色散点表示预测结果。
- **hold on**：
  - 保持当前图形，允许在同一图形上绘制多条曲线。
- **plot(xlim, ylim, '--k')**：
  - 绘制理想预测线，即真实值等于预测值的对角线，使用黑色虚线表示。
- **xlabel('训练集真实值')** 和 **ylabel('训练集预测值')**：
  - 设置X轴和Y轴的标签为“训练集真实值”和“训练集预测值”。
- **xlim([min(T_train) max(T_train)])** 和 **ylim([min(T_sim1) max(T_sim1)])**：
  - 设置X轴和Y轴的显示范围为数据的最小值和最大值。
- **title('训练集预测值 vs. 训练集真实值')**：
  - 设置图形的标题为“训练集预测值 vs. 训练集真实值”。

##### 绘制测试集散点图

```matlab
figure
scatter(T_test, T_sim2, sz, c)               % 绘制测试集真实值与预测值的散点图，蓝色散点表示预测结果
hold on                                       % 保持当前图形，允许在同一图形上绘制多条曲线
plot(xlim, ylim, '--k')                      % 绘制理想预测线（真实值等于预测值的对角线），使用黑色虚线表示
xlabel('测试集真实值');                         % 设置X轴标签为“测试集真实值”
ylabel('测试集预测值');                         % 设置Y轴标签为“测试集预测值”
xlim([min(T_test) max(T_test)])                % 设置X轴的显示范围为[最小真实值, 最大真实值]
ylim([min(T_sim2) max(T_sim2)])                % 设置Y轴的显示范围为[最小预测值, 最大预测值]
title('测试集预测值 vs. 测试集真实值')             % 设置图形的标题为“测试集预测值 vs. 测试集真实值”
```

- **figure**：创建新的图形窗口。
- **scatter(T_test, T_sim2, sz, c)**：
  - 使用`scatter`函数绘制测试集真实值`T_test`与预测值`t_sim2`的散点图，蓝色散点表示预测结果。
- **hold on**：
  - 保持当前图形，允许在同一图形上绘制多条曲线。
- **plot(xlim, ylim, '--k')**：
  - 绘制理想预测线，即真实值等于预测值的对角线，使用黑色虚线表示。
- **xlabel('测试集真实值')** 和 **ylabel('测试集预测值')**：
  - 设置X轴和Y轴的标签为“测试集真实值”和“测试集预测值”。
- **xlim([min(T_test) max(T_test)])** 和 **ylim([min(T_sim2) max(T_sim2)])**：
  - 设置X轴和Y轴的显示范围为数据的最小值和最大值。
- **title('测试集预测值 vs. 测试集真实值')**：
  - 设置图形的标题为“测试集预测值 vs. 测试集真实值”。

---

### 代码使用注意事项

1. **数据集格式**：
   - **时间序列数据**：确保`数据集.xlsx`中的数据为单列时间序列数据，表示时间序列的连续值。
   - **数据顺序**：时间序列数据应按照时间顺序排列，确保数据的时间依赖关系。

2. **参数调整**：
   - **延时步长（kim）**：通过`kim = 15`设定，表示使用15个历史数据点作为输入特征。根据时间序列的特性和周期性调整延时步长，步长过大可能导致模型复杂度增加，步长过小可能导致模型捕捉不到足够的时间依赖信息。
   - **预测步长（zim）**：通过`zim = 1`设定，表示预测当前点之后的1个时间点的值。根据实际需求调整预测步长，适用于单步预测或多步预测。
   - **训练集比例（num_size）**：通过`num_size = 0.7`设定，表示70%的数据用于训练，30%的数据用于测试。根据数据集大小和分布调整训练集比例，确保训练集和测试集具有代表性。
   - **隐藏层神经元数量（hiddennum）**：通过`hiddennum = 5`设定隐藏层节点个数。根据数据的复杂度和特征数量调整隐藏层神经元数量，神经元过少可能导致欠拟合，神经元过多可能导致过拟合。
   - **PSO算法参数**：
     - **学习因子（c1, c2）**：设定为4.494，控制粒子向个体最佳和全局最佳位置的移动速度。
     - **最大迭代次数（maxgen）**：设定为30，根据问题的复杂度和计算资源调整PSO算法的最大迭代次数。
     - **种群规模（sizepop）**：设定为5，表示每代有5个粒子。较大的种群规模可能提升搜索能力，但增加计算成本。
     - **速度和位置边界（Vmax, Vmin, popmax, popmin）**：根据具体问题调整粒子的速度和位置边界，确保搜索空间合理。

3. **环境要求**：
   - **MATLAB版本**：确保使用的MATLAB版本支持`newff`、`mapminmax`、`train`、`sim`等函数。需要安装MATLAB的 Neural Network Toolbox。
   - **工具箱**：
     - **Neural Network Toolbox**：支持使用BP神经网络相关函数，如`newff`、`train`、`sim`等。

4. **性能优化**：
   - **数据预处理**：
     - **归一化**：通过`mapminmax`函数对输入数据和目标变量进行归一化，提升模型训练速度和稳定性。
     - **降维**：如果输入特征过多，可以考虑使用主成分分析（PCA）等降维方法，减少特征数量，提升模型训练效率和性能。
   - **模型参数优化**：
     - **隐藏层神经元数量**：根据数据的复杂度和特征数量调整隐藏层神经元数量，优化模型的特征提取能力和拟合能力。
     - **学习率调整**：通过调整BP网络的学习率，提升模型的收敛速度和稳定性。
     - **PSO算法参数调整**：根据优化问题的特性调整PSO算法的参数，如学习因子、种群规模和最大迭代次数，提升优化效果。

5. **结果验证**：
   - **交叉验证**：采用k折交叉验证方法评估模型的稳定性和泛化能力，避免因数据划分偶然性导致的性能波动。
   - **多次运行**：由于PSO-BP模型对初始粒子和优化过程敏感，建议多次运行模型，取平均性能指标，以获得更稳定的评估结果。
   - **模型对比**：将PSO-BP时序预测模型与其他预测模型（如ARIMA、LSTM、ELM等）进行对比，评估不同模型在相同数据集上的表现差异。

6. **性能指标理解**：
   - **决定系数（R²）**：衡量模型对数据的拟合程度，值越接近1表示模型解释变量变异的能力越强。
   - **平均绝对误差（MAE）**：表示预测值与真实值之间的平均绝对差异，值越小表示模型性能越好。
   - **平均偏差误差（MBE）**：表示预测值与真实值之间的平均差异，正值表示模型倾向于高估，负值表示模型倾向于低估。
   - **平均绝对百分比误差（MAPE）**：表示预测值与真实值之间的平均绝对百分比差异，适用于评估相对误差。
   - **均方根误差（RMSE）**：表示预测值与真实值之间的平方差的平均值的平方根，值越小表示模型性能越好。

7. **模型分析与可视化**：
   - **网络结构分析**：通过`analyzeNetwork`函数可视化和分析BP网络的结构和参数，了解网络的层次结构和连接权重。
   - **训练过程监控**：通过绘制适应度变化曲线，了解PSO算法优化过程中的适应度值变化趋势和收敛情况。
   - **预测结果对比图**：通过绘制训练集和测试集的真实值与预测值对比图，直观展示模型的预测效果。
   - **散点图**：通过绘制真实值与预测值的散点图，评估模型的拟合能力和误差分布。
   - **误差分析**：通过计算并分析RMSE、R²、MAE、MBE、MAPE等指标，全面评估模型的性能和预测准确性。

8. **代码适应性**：
   - **模型参数调整**：根据实际数据和任务需求，调整PSO-BP模型的参数（如隐藏层神经元数量、学习率等），优化模型性能。
   - **数据格式匹配**：确保输入数据的格式与BP神经网络和PSO算法的要求一致。输入数据应为行样本、列特征的矩阵，目标变量为列向量。
   - **特征处理**：如果输入数据包含类别特征，需先进行数值编码转换，确保所有特征均为数值型数据。

通过理解和应用上述PSO-BP时序预测模型，用户可以有效地处理各种时间序列预测任务，充分发挥PSO算法在全局优化和BP神经网络在高效学习方面的优势，提升模型的预测准确性和鲁棒性。
