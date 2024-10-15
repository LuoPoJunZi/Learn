clc;
clear;
close all;

%% 问题定义 (Problem Definition)

CostFunction = @(x) MOP2(x);  % 目标函数，用于多目标优化

nVar = 3;  % 决策变量的数量

VarSize = [1 nVar];  % 决策变量矩阵的大小

VarMin = -5;  % 决策变量的下界
VarMax = 5;   % 决策变量的上界

% 目标函数数量
nObj = numel(CostFunction(unifrnd(VarMin, VarMax, VarSize)));

%% NSGA-II 参数设置 (NSGA-II Parameters)

MaxIt = 100;  % 最大迭代次数

nPop = 50;  % 种群大小

pCrossover = 0.7;  % 交叉操作的概率
nCrossover = 2 * round(pCrossover * nPop / 2);  % 交叉生成的子代数量

pMutation = 0.4;  % 突变操作的概率
nMutation = round(pMutation * nPop);  % 突变的个体数量

mu = 0.02;  % 突变率

sigma = 0.1 * (VarMax - VarMin);  % 突变步长

%% 初始化 (Initialization)

% 定义个体结构
empty_individual.Position = [];  % 决策变量的位置
empty_individual.Cost = [];  % 目标函数值
empty_individual.Rank = [];  % 支配等级
empty_individual.DominationSet = [];  % 支配集
empty_individual.DominatedCount = [];  % 被支配计数
empty_individual.CrowdingDistance = [];  % 拥挤距离

pop = repmat(empty_individual, nPop, 1);  % 初始化种群

% 初始化种群个体
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);  % 随机生成个体位置
    pop(i).Cost = CostFunction(pop(i).Position);  % 计算个体的目标函数值
end

% 非支配排序
[pop, F] = NonDominatedSorting(pop);

% 计算拥挤距离
pop = CalcCrowdingDistance(pop, F);

% 对种群进行排序
[pop, F] = SortPopulation(pop);

%% NSGA-II 主循环 (NSGA-II Main Loop)

for it = 1:MaxIt
    % 交叉操作 (Crossover)
    popc = repmat(empty_individual, nCrossover/2, 2);  % 初始化子代种群
    for k = 1:nCrossover/2
        % 随机选择两个父代
        i1 = randi([1 nPop]);
        p1 = pop(i1);
        
        i2 = randi([1 nPop]);
        p2 = pop(i2);
        
        % 执行交叉操作，生成两个子代
        [popc(k,1).Position, popc(k,2).Position] = Crossover(p1.Position, p2.Position);
        
        % 计算子代的目标函数值
        popc(k,1).Cost = CostFunction(popc(k,1).Position);
        popc(k,2).Cost = CostFunction(popc(k,2).Position);
    end
    popc = popc(:);  % 将子代矩阵转为列向量
    
    % 突变操作 (Mutation)
    popm = repmat(empty_individual, nMutation, 1);  % 初始化突变种群
    for k = 1:nMutation
        i = randi([1 nPop]);  % 随机选择一个个体进行突变
        p = pop(i);
        
        % 执行突变操作
        popm(k).Position = Mutate(p.Position, mu, sigma);
        
        % 计算突变后个体的目标函数值
        popm(k).Cost = CostFunction(popm(k).Position);
    end
    
    % 合并种群 (Merge)
    pop = [pop; popc; popm];  % 合并当前种群、交叉产生的子代和突变个体
    
    % 非支配排序
    [pop, F] = NonDominatedSorting(pop);

    % 计算拥挤距离
    pop = CalcCrowdingDistance(pop, F);

    % 对种群进行排序
    [pop, F] = SortPopulation(pop);
    
    % 截断种群 (Truncate)
    pop = pop(1:nPop);  % 保持种群大小为nPop
    
    % 再次进行非支配排序和拥挤距离计算
    [pop, F] = NonDominatedSorting(pop);
    pop = CalcCrowdingDistance(pop, F);
    [pop, F] = SortPopulation(pop);
    
    % 保存当前前沿F1（即最优解集）
    F1 = pop(F{1});
    
    % 显示迭代信息
    disp(['Iteration ' num2str(it) ': Number of F1 Members = ' num2str(numel(F1))]);
    
    % 绘制F1的目标函数值
    figure(1);
    PlotCosts(F1);
    pause(0.01);  % 短暂暂停，用于实时显示图像
end

%% 结果展示 (Results)
