clc;        % 清除命令窗口
clear;      % 清除工作区变量
close all;  % 关闭所有图形窗口

%% 问题定义

% 定义目标函数，这里使用MOP2
CostFunction = @(x) MOP2(x);

nVar = 3;             % 决策变量的数量
VarSize = [nVar 1];   % 决策变量的矩阵尺寸

VarMin = 0;           % 决策变量的下界
VarMax = 1;           % 决策变量的上界

% 计算目标的数量，通过对一个随机解计算目标数
nObj = numel(CostFunction(unifrnd(VarMin, VarMax, VarSize)));

%% PESA-II 设置

MaxIt = 100;        % 最大迭代次数
nPop = 50;          % 种群大小
nArchive = 50;      % 存档大小
nGrid = 7;          % 每个维度的网格数量
InflationFactor = 0.1;  % 网格膨胀因子

beta_deletion = 1;  % 删除操作的参数
beta_selection = 2; % 选择操作的参数

pCrossover = 0.5;    % 交叉概率
nCrossover = round(pCrossover * nPop / 2) * 2;  % 交叉操作的次数，确保为偶数

pMutation = 1 - pCrossover;  % 变异概率
nMutation = nPop - nCrossover;  % 变异操作的次数

% 交叉操作的参数
crossover_params.gamma = 0.15;
crossover_params.VarMin = VarMin;
crossover_params.VarMax = VarMax;

% 变异操作的参数
mutation_params.h = 0.3;
mutation_params.VarMin = VarMin;
mutation_params.VarMax = VarMax;

%% 初始化

% 定义一个空的个体结构
empty_individual.Position = [];
empty_individual.Cost = [];
empty_individual.IsDominated = [];
empty_individual.GridIndex = [];

% 初始化种群
pop = repmat(empty_individual, nPop, 1);

for i = 1:nPop
    % 随机初始化个体的位置
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    % 计算个体的目标值
    pop(i).Cost = CostFunction(pop(i).Position);
end

% 初始化存档为空
archive = [];

%% 主循环

for it = 1:MaxIt
    
    % 确定种群中每个个体是否被支配
    pop = DetermineDomination(pop);
    
    % 提取非被支配的个体
    ndpop = pop(~[pop.IsDominated]);
    
    % 将非被支配的个体添加到存档中
    archive = [archive
               ndpop]; %#ok
    
    % 确定存档中每个个体是否被支配
    archive = DetermineDomination(archive);
    
    % 提取非被支配的存档个体
    archive = archive(~[archive.IsDominated]);
    
    % 创建网格并将存档中的个体分配到网格中
    [archive, grid] = CreateGrid(archive, nGrid, InflationFactor);
    
    % 如果存档大小超过限制，进行截断
    if numel(archive) > nArchive
        E = numel(archive) - nArchive;  % 需要删除的个体数量
        archive = TruncatePopulation(archive, grid, E, beta_deletion);
        % 重新创建网格
        [archive, grid] = CreateGrid(archive, nGrid, InflationFactor);
    end
    
    % Pareto 前沿
    PF = archive;
    
    % 绘制当前的Pareto前沿
    figure(1);
    PlotCosts(PF);
    pause(0.01);
    
    % 显示当前迭代的信息
    disp(['Iteration ' num2str(it) ': Number of PF Members = ' num2str(numel(PF))]);
    
    % 检查是否达到最大迭代次数
    if it >= MaxIt
        break;
    end
    
    %% 交叉操作
    popc = repmat(empty_individual, nCrossover / 2, 2);  % 交叉生成的子代个体
    for c = 1:nCrossover / 2
        % 从存档中选择两个父代个体
        p1 = SelectFromPopulation(archive, grid, beta_selection);
        p2 = SelectFromPopulation(archive, grid, beta_selection);
        
        % 执行交叉操作，生成两个子代
        [popc(c, 1).Position, popc(c, 2).Position] = Crossover(p1.Position, ...
                                                                 p2.Position, ...
                                                                 crossover_params);
        
        % 计算子代的目标值
        popc(c, 1).Cost = CostFunction(popc(c, 1).Position);
        popc(c, 2).Cost = CostFunction(popc(c, 2).Position);
    end
    popc = popc(:);  % 将交叉生成的子代展平成一维数组
    
    %% 变异操作
    popm = repmat(empty_individual, nMutation, 1);  % 变异生成的个体
    for m = 1:nMutation
        % 从存档中选择一个父代个体
        p = SelectFromPopulation(archive, grid, beta_selection);
        
        % 执行变异操作，生成子代
        popm(m).Position = Mutate(p.Position, mutation_params);
        
        % 计算子代的目标值
        popm(m).Cost = CostFunction(popm(m).Position);
    end
    
    % 将交叉和变异生成的子代添加到种群中
    pop = [popc
           popm];
             
end

%% 结果展示

disp(' ');

% 提取Pareto前沿的目标值
PFC = [PF.Cost];
for j = 1:size(PFC, 1)
    disp(['Objective #' num2str(j) ':']);
    disp(['      Min = ' num2str(min(PFC(j, :)))]);
    disp(['      Max = ' num2str(max(PFC(j, :)))]);
    disp(['    Range = ' num2str(max(PFC(j, :)) - min(PFC(j, :)))]);
    disp(['    St.D. = ' num2str(std(PFC(j, :)))]);
    disp(['     Mean = ' num2str(mean(PFC(j, :)))]);
    disp(' ');
end
