% MOEA-D算法的实现
clc;                % 清除命令行
clear;              % 清除工作区
close all;         % 关闭所有图形窗口

%% 问题定义

CostFunction = @(x) MOP2(x);  % 目标函数，使用MOP2函数进行评估

nVar = 3;             % 决策变量的数量

VarSize = [nVar 1];   % 决策变量矩阵的大小

VarMin = 0;           % 决策变量的下界
VarMax = 1;           % 决策变量的上界

nObj = numel(CostFunction(unifrnd(VarMin, VarMax, VarSize)));  % 目标函数的数量


%% MOEA/D设置

MaxIt = 100;  % 最大迭代次数

nPop = 50;    % 种群规模（子问题数量）

nArchive = 50;  % 存档数量

T = max(ceil(0.15 * nPop), 2);    % 邻居数量
T = min(max(T, 2), 15);            % 确保邻居数量在合理范围内

crossover_params.gamma = 0.5;  % 交叉参数
crossover_params.VarMin = VarMin;  % 决策变量下界
crossover_params.VarMax = VarMax;  % 决策变量上界

%% 初始化

% 创建子问题
sp = CreateSubProblems(nObj, nPop, T);

% 空个体（作为模板）
empty_individual.Position = [];  % 个体位置
empty_individual.Cost = [];      % 代价
empty_individual.g = [];          % 分解代价
empty_individual.IsDominated = [];  % 支配状态

% 初始化目标点
% z = inf(nObj, 1);  % 可选，未使用
z = zeros(nObj, 1);  % 初始化目标点为零

% 创建初始种群（随机初始化）
pop = repmat(empty_individual, nPop, 1);  % 复制空个体
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);  % 随机分布
    pop(i).Cost = CostFunction(pop(i).Position);  % 计算代价
    z = min(z, pop(i).Cost);  % 更新理想点
end

% 计算每个个体的分解代价
for i = 1:nPop
    pop(i).g = DecomposedCost(pop(i), z, sp(i).lambda);
end

% 确定种群的支配状态
pop = DetermineDomination(pop);

% 初始化估计的Pareto前沿
EP = pop(~[pop.IsDominated]);

%% 主循环

for it = 1:MaxIt
    for i = 1:nPop
        
        % 选择个体进行重组（交叉操作）
        K = randsample(T, 2);  % 随机选择两个邻居
        
        j1 = sp(i).Neighbors(K(1));  % 第一个邻居的索引
        p1 = pop(j1);  % 第一个邻居的个体
        
        j2 = sp(i).Neighbors(K(2));  % 第二个邻居的索引
        p2 = pop(j2);  % 第二个邻居的个体
        
        y = empty_individual;  % 创建一个新的空个体
        y.Position = Crossover(p1.Position, p2.Position, crossover_params);  % 交叉生成新个体
        
        y.Cost = CostFunction(y.Position);  % 计算新个体的代价
        
        z = min(z, y.Cost);  % 更新理想点
        
        % 更新邻居个体
        for j = sp(i).Neighbors
            y.g = DecomposedCost(y, z, sp(j).lambda);  % 计算新个体的分解代价
            if y.g <= pop(j).g
                pop(j) = y;  % 更新邻居个体
            end
        end
        
    end
    
    % 确定种群的支配状态
    pop = DetermineDomination(pop);
    
    ndpop = pop(~[pop.IsDominated]);  % 非支配个体
    
    EP = [EP; ndpop];  % 更新Pareto前沿
    
    EP = DetermineDomination(EP);  % 确定Pareto前沿的支配状态
    EP = EP(~[EP.IsDominated]);  % 仅保留非支配个体
    
    % 如果估计的Pareto前沿超过存档限制，则随机删除部分个体
    if numel(EP) > nArchive
        Extra = numel(EP) - nArchive;  % 计算需要删除的个体数量
        ToBeDeleted = randsample(numel(EP), Extra);  % 随机选择要删除的个体
        EP(ToBeDeleted) = [];  % 删除个体
    end
    
    % 绘制Pareto前沿
    figure(1);
    PlotCosts(EP);  % 绘制当前Pareto前沿
    pause(0.01);  % 暂停以便可视化更新

    % 显示当前迭代的信息
    disp(['Iteration ' num2str(it) ': Number of Pareto Solutions = ' num2str(numel(EP))]);
    
end

%% 结果输出

disp(' ');

EPC = [EP.Cost];  % 提取代价信息
for j = 1:nObj
    
    disp(['Objective #' num2str(j) ':']);
    disp(['      Min = ' num2str(min(EPC(j, :)))]);  % 输出最小值
    disp(['      Max = ' num2str(max(EPC(j, :)))]);  % 输出最大值
    disp(['    Range = ' num2str(max(EPC(j, :)) - min(EPC(j, :)))]);  % 输出范围
    disp(['    St.D. = ' num2str(std(EPC(j, :)))]);  % 输出标准差
    disp(['     Mean = ' num2str(mean(EPC(j, :)))]);  % 输出均值
    disp(' ');
    
end
