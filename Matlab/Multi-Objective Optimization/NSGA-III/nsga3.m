% K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm 
% Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
% Problems With Box Constraints,"
% IEEE Transactions on Evolutionary Computation,
% vol. 18, no. 4, pp. 577-601, Aug. 2014.

clc;  % 清除命令窗口
clear;  % 清除工作区
close all;  % 关闭所有图形窗口

%% 问题定义

CostFunction = @(x) MOP2(x);  % 成本函数，使用 MOP2 作为目标函数

nVar = 5;    % 决策变量数量

VarSize = [1 nVar]; % 决策变量矩阵的大小

VarMin = -1;   % 决策变量的下界
VarMax = 1;    % 决策变量的上界

% 目标函数数量
nObj = numel(CostFunction(unifrnd(VarMin, VarMax, VarSize)));

%% NSGA-III 参数设置

% 生成参考点
nDivision = 10;  % 每个目标的划分数
Zr = GenerateReferencePoints(nObj, nDivision);  % 生成参考点矩阵

MaxIt = 50;  % 最大迭代次数

nPop = 80;  % 种群大小

pCrossover = 0.5;       % 交叉概率
nCrossover = 2 * round(pCrossover * nPop / 2); % 父代数量（子代数量）

pMutation = 0.5;       % 变异概率
nMutation = round(pMutation * nPop);  % 突变个体数量

mu = 0.02;     % 变异率

sigma = 0.1 * (VarMax - VarMin); % 变异步长


%% 收集参数

params.nPop = nPop;  % 种群大小
params.Zr = Zr;  % 参考点矩阵
params.nZr = size(Zr, 2);  % 参考点数量
params.zmin = [];  % 理想点的初始化
params.zmax = [];  % 最优点的初始化
params.smin = [];  % 其他参数初始化

%% 初始化

disp('开始 NSGA-III ...');

% 创建空个体结构
empty_individual.Position = [];  % 个体位置
empty_individual.Cost = [];  % 个体成本
empty_individual.Rank = [];  % 个体排名
empty_individual.DominationSet = [];  % 支配集
empty_individual.DominatedCount = [];  % 被支配计数
empty_individual.NormalizedCost = [];  % 归一化成本
empty_individual.AssociatedRef = [];  % 关联参考点
empty_individual.DistanceToAssociatedRef = [];  % 与关联参考点的距离

% 初始化种群
pop = repmat(empty_individual, nPop, 1);  % 复制空个体
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);  % 随机初始化个体位置
    pop(i).Cost = CostFunction(pop(i).Position);  % 计算个体成本
end

% 对种群进行排序和选择
[pop, F, params] = SortAndSelectPopulation(pop, params);


%% NSGA-III 主循环

for it = 1:MaxIt
 
    % 交叉
    popc = repmat(empty_individual, nCrossover / 2, 2);  % 存放子代的结构
    for k = 1:nCrossover / 2
        i1 = randi([1 nPop]);  % 随机选择第一个父代
        p1 = pop(i1);  % 获取父代个体

        i2 = randi([1 nPop]);  % 随机选择第二个父代
        p2 = pop(i2);  % 获取父代个体

        % 执行交叉操作
        [popc(k, 1).Position, popc(k, 2).Position] = Crossover(p1.Position, p2.Position);

        % 计算子代的成本
        popc(k, 1).Cost = CostFunction(popc(k, 1).Position);
        popc(k, 2).Cost = CostFunction(popc(k, 2).Position);
    end
    popc = popc(:);  % 转换为列向量

    % 变异
    popm = repmat(empty_individual, nMutation, 1);  % 存放突变个体的结构
    for k = 1:nMutation
        i = randi([1 nPop]);  % 随机选择个体进行变异
        p = pop(i);  % 获取个体

        % 执行变异操作
        popm(k).Position = Mutate(p.Position, mu, sigma);

        % 计算突变个体的成本
        popm(k).Cost = CostFunction(popm(k).Position);
    end

    % 合并种群
    pop = [pop; popc; popm]; %#ok
    
    % 对合并后的种群进行排序和选择
    [pop, F, params] = SortAndSelectPopulation(pop, params);
    
    % 存储 F1 前沿
    F1 = pop(F{1});

    % 显示迭代信息
    disp(['迭代 ' num2str(it) ': F1 中的成员数量 = ' num2str(numel(F1))]);

    % 绘制 F1 成本
    figure(1);
    PlotCosts(F1);  % 绘制前沿成本
    pause(0.01);  % 暂停以便于可视化
 
end

%% 结果

disp(['最终迭代: F1 中的成员数量 = ' num2str(numel(F1))]);
disp('优化结束。');
