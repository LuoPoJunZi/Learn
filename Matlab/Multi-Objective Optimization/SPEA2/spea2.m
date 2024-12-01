% spea2.m
% 强度Pareto进化算法2 (SPEA2) 的主程序

clc;        % 清除命令行窗口
clear;      % 清除工作区变量
close all;  % 关闭所有图形窗口

%% 问题定义

CostFunction = @(x) ZDT(x);    % 目标函数句柄，这里使用ZDT函数

nVar = 30;                     % 决策变量的数量

VarSize = [nVar 1];            % 决策变量矩阵的大小

VarMin = 0;                    % 决策变量的下界
VarMax = 1;                    % 决策变量的上界

%% SPEA2 设置

MaxIt = 200;                   % 最大迭代次数

nPop = 50;                     % 种群大小

nArchive = 50;                 % 存档大小

K = round(sqrt(nPop + nArchive));  % K近邻参数，用于环境选择

pCrossover = 0.7;              % 交叉概率
nCrossover = round(pCrossover * nPop / 2) * 2;  % 交叉操作生成的后代数量，确保为偶数

pMutation = 1 - pCrossover;    % 变异概率
nMutation = nPop - nCrossover; % 变异操作生成的后代数量

% 交叉操作的参数
crossover_params.gamma = 0.1;
crossover_params.VarMin = VarMin;
crossover_params.VarMax = VarMax;

% 变异操作的参数
mutation_params.h = 0.2;
mutation_params.VarMin = VarMin;
mutation_params.VarMax = VarMax;

%% 初始化

% 定义一个空的个体结构体，包含位置、成本以及其他SPEA2需要的字段
empty_individual.Position = [];
empty_individual.Cost = [];
empty_individual.S = [];
empty_individual.R = [];
empty_individual.sigma = [];
empty_individual.sigmaK = [];
empty_individual.D = [];
empty_individual.F = [];

% 初始化种群
pop = repmat(empty_individual, nPop, 1);
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);   % 随机初始化个体位置
    pop(i).Cost = CostFunction(pop(i).Position);         % 计算个体的成本（目标函数值）
end

archive = [];  % 初始化存档为空

%% 主循环

for it = 1:MaxIt
    Q = [pop
         archive];  % 合并种群和存档
    
    nQ = numel(Q);   % 合并后的种群大小
    
    dom = false(nQ, nQ);  % 初始化支配关系矩阵
    
    % 初始化每个个体的S值为0
    for i = 1:nQ
        Q(i).S = 0;
    end
    
    % 计算支配关系
    for i = 1:nQ
        for j = i+1:nQ
            if Dominates(Q(i), Q(j))
                Q(i).S = Q(i).S + 1;  % Q(i)支配Q(j)
                dom(i, j) = true;
            elseif Dominates(Q(j), Q(i))
                Q(j).S = Q(j).S + 1;  % Q(j)支配Q(i)
                dom(j, i) = true;
            end
        end
    end
    
    S = [Q.S];  % 获取所有个体的S值
    for i = 1:nQ
        Q(i).R = sum(S(dom(:, i)));  % 计算个体i的R值，表示被支配的次数
    end
    
    Z = [Q.Cost]';  % 获取所有个体的目标函数值
    SIGMA = pdist2(Z, Z, 'seuclidean');  % 计算标准化欧氏距离矩阵
    SIGMA = sort(SIGMA);                % 对距离进行排序
    for i = 1:nQ
        Q(i).sigma = SIGMA(:, i);        % 获取个体i的距离向量
        Q(i).sigmaK = Q(i).sigma(K);     % 获取第K近邻的距离
        Q(i).D = 1 / (Q(i).sigmaK + 2);  % 计算密度估计
        Q(i).F = Q(i).R + Q(i).D;        % 计算适应度值F
    end
    
    nND = sum([Q.R] == 0);  % 计算非支配解的数量
    if nND <= nArchive
        F = [Q.F];
        [F, SO] = sort(F);          % 按适应度值排序
        Q = Q(SO);                   % 按适应度排序后的个体
        archive = Q(1:min(nArchive, nQ));  % 更新存档
    else
        SIGMA = SIGMA(:, [Q.R] == 0);    % 仅考虑非支配解的距离
        archive = Q([Q.R] == 0);         % 更新存档为所有非支配解
        
        k = 2;
        while numel(archive) > nArchive
            % 找到距离第k层的个体中最拥挤的个体
            while min(SIGMA(k, :)) == max(SIGMA(k, :)) && k < size(SIGMA, 1)
                k = k + 1;
            end
            
            [~, j] = min(SIGMA(k, :));  % 找到最小距离对应的个体索引
            
            archive(j) = [];              % 从存档中移除该个体
            SIGMA(:, j) = [];             % 更新距离矩阵
        end
    end
    
    PF = archive([archive.R] == 0);  % 近似Pareto前沿
    
    % 绘制Pareto前沿
    figure(1);
    PlotCosts(PF);
    pause(0.01);  % 暂停以更新图形
    
    % 显示当前迭代的信息
    disp(['迭代 ' num2str(it) ': Pareto前沿成员数量 = ' num2str(numel(PF))]);
    
    if it >= MaxIt
        break;  % 达到最大迭代次数，退出循环
    end
    
    %% 交叉操作
    popc = repmat(empty_individual, nCrossover / 2, 2);  % 初始化交叉后代矩阵
    for c = 1:nCrossover / 2
        % 选择两个父代个体
        p1 = BinaryTournamentSelection(archive, [archive.F]);
        p2 = BinaryTournamentSelection(archive, [archive.F]);
        
        % 进行交叉，生成两个子代
        [popc(c, 1).Position, popc(c, 2).Position] = Crossover(p1.Position, p2.Position, crossover_params);
        
        % 计算子代的成本
        popc(c, 1).Cost = CostFunction(popc(c, 1).Position);
        popc(c, 2).Cost = CostFunction(popc(c, 2).Position);
    end
    popc = popc(:);  % 将后代矩阵转化为向量
    
    %% 变异操作
    popm = repmat(empty_individual, nMutation, 1);  % 初始化变异后代矩阵
    for m = 1:nMutation
        % 选择一个父代个体
        p = BinaryTournamentSelection(archive, [archive.F]);
        
        % 进行变异，生成子代
        popm(m).Position = Mutate(p.Position, mutation_params);
        
        % 计算子代的成本
        popm(m).Cost = CostFunction(popm(m).Position);
    end
    
    %% 创建新种群
    pop = [popc
           popm];  % 新的种群由交叉和变异生成的后代组成
end

%% 结果展示

disp(' ');

PFC = [PF.Cost];  % 获取Pareto前沿的目标函数值
for j = 1:size(PFC, 1)
    disp(['目标 #' num2str(j) ':']);
    disp(['      最小值 = ' num2str(min(PFC(j, :)))]);
    disp(['      最大值 = ' num2str(max(PFC(j, :)))]);
    disp(['    范围 = ' num2str(max(PFC(j, :)) - min(PFC(j, :)))]);
    disp(['    标准差 = ' num2str(std(PFC(j, :)))]);
    disp(['     均值 = ' num2str(mean(PFC(j, :)))]);
    disp(' ');
end
