# MOEA-D（基于分解的多目标进化算法）

MOEA-D（基于分解的多目标进化算法）是一种用于多目标优化的进化算法，其核心思想是将多目标优化问题分解为多个单目标优化问题。下面是MOEA-D算法的几个主要特点和步骤：

### 主要特点

1. **分解策略**：
   - 将多目标优化问题通过权重向量（lambda）划分为多个子问题。每个子问题对应一个权重向量，代表目标空间中的一个方向。

2. **邻域交互**：
   - 每个子问题与其邻域内的其他子问题相互影响，这有助于保持种群的多样性并促进优良解的传播。

3. **适应度评估**：
   - 通过计算每个个体与理想点的距离，评估其适应度。采用加权绝对差的最大值作为个体的代价。

4. **多样性维护**：
   - 通过选择、交叉和变异等操作，保持种群的多样性，避免早熟收敛。

### 算法步骤

1. **初始化**：
   - 随机生成初始种群和对应的权重向量，确定邻域关系。

2. **循环迭代**：
   - 在每次迭代中，通过交叉和变异等操作生成新个体。
   - 更新每个个体的代价，使用分解代价函数评估解的质量。
   - 通过支配关系确定哪些个体被优先保留。

3. **更新Pareto前沿**：
   - 在每次迭代后，更新估计的Pareto前沿，并在必要时进行个体的选择和替换。

4. **终止条件**：
   - 当达到预设的最大迭代次数或满足其他停止条件时，算法结束，返回Pareto前沿的解集。

### 应用场景

MOEA-D算法广泛应用于工程优化、资源分配、路径规划等领域，尤其是在多个相互冲突的目标之间寻求平衡时表现出色。


## `CreateSubProblems.m` 文件代码：

```matlab
% 创建子问题的函数
% 输入参数:
% nObj - 目标函数的数量
% nPop - 种群数量
% T - 每个子问题的邻居数量

function sp = CreateSubProblems(nObj, nPop, T)

    % 初始化空的子问题结构体
    empty_sp.lambda = [];       % 存储子问题的权重向量
    empty_sp.Neighbors = [];    % 存储邻居子问题的索引

    % 使用空结构体复制生成种群数量的子问题
    sp = repmat(empty_sp, nPop, 1);
    
    % theta = linspace(0, pi/2, nPop); % 可选：均匀分布的角度，未使用

    for i = 1:nPop
        % 随机生成权重向量并进行归一化
        lambda = rand(nObj, 1);   
        lambda = lambda / norm(lambda);  % 归一化处理
        sp(i).lambda = lambda;  % 将权重向量存储到子问题中
        
        % 可选：使用角度生成权重向量，未使用
        % sp(i).lambda = [cos(theta(i))
        %                  sin(theta(i))];
    end

    % 从子问题中提取权重向量并计算距离矩阵
    LAMBDA = [sp.lambda]';  
    D = pdist2(LAMBDA, LAMBDA);  % 计算权重向量之间的欧氏距离
    
    for i = 1:nPop
        % 对距离进行排序，获取最近的T个邻居
        [~, SO] = sort(D(i, :));
        sp(i).Neighbors = SO(1:T);  % 存储邻居索引
    end

end
```


## `Crossover.m` 文件代码：

```matlab
% 交叉操作函数
% 输入参数:
% x1 - 第一个父代个体
% x2 - 第二个父代个体
% params - 参数结构体，包含交叉参数和变量范围

function y = Crossover(x1, x2, params)

    % 从参数中提取交叉相关参数
    gamma = params.gamma;       % 交叉幅度参数
    VarMin = params.VarMin;     % 变量的最小值
    VarMax = params.VarMax;     % 变量的最大值
    
    % 生成交叉因子，范围在[-gamma, 1 + gamma]之间
    alpha = unifrnd(-gamma, 1 + gamma, size(x1));
    
    % 进行线性组合生成子代个体
    y = alpha .* x1 + (1 - alpha) .* x2;

    % 确保生成的个体在变量范围内
    y = min(max(y, VarMin), VarMax);
    
end
```

## `DecomposedCost.m` 文件代码：

```matlab
% 计算分解代价的函数
% 输入参数:
% individual - 当前个体，可以是代价向量或结构体
% z - 理想点（目标函数的真实值）
% lambda - 权重向量，用于分解计算

function g = DecomposedCost(individual, z, lambda)

    % 检查个体是否包含代价值
    if isfield(individual, 'Cost')
        fx = individual.Cost;  % 从结构体中提取代价值
    else
        fx = individual;  % 如果不是结构体，直接使用个体作为代价
    end
    
    % 计算分解代价，使用加权绝对差的最大值
    g = max(lambda .* abs(fx - z));  % 计算当前个体与理想点的代价

end
```

## `DetermineDomination.m` 文件代码：

```matlab
% 确定种群中个体的支配关系的函数
% 输入参数:
% pop - 种群数组，包含多个个体

function pop = DetermineDomination(pop)

    nPop = numel(pop);  % 获取种群中个体的数量

    % 初始化每个个体的支配状态
    for i = 1:nPop
        pop(i).IsDominated = false;  % 默认未被支配
    end
    
    % 进行双重循环比较每对个体
    for i = 1:nPop
        for j = i + 1:nPop
            if Dominates(pop(i), pop(j))
                % 如果个体i支配个体j
                pop(j).IsDominated = true;  % 将j标记为被支配
                
            elseif Dominates(pop(j), pop(i))
                % 如果个体j支配个体i
                pop(i).IsDominated = true;  % 将i标记为被支配
                
            end
        end
    end

end
```

## `Dominates.m` 文件代码：

```matlab
% 判断个体x是否支配个体y的函数
% 输入参数:
% x - 被比较的个体
% y - 另一个被比较的个体

function b = Dominates(x, y)

    % 如果个体x是结构体，则提取代价值
    if isfield(x, 'Cost')
        x = x.Cost;  % 获取个体x的代价
    end

    % 如果个体y是结构体，则提取代价值
    if isfield(y, 'Cost')
        y = y.Cost;  % 获取个体y的代价
    end
    
    % 判断x是否支配y
    % x支配y当且仅当x在所有目标上都不大于y，并且至少在一个目标上严格小于y
    b = all(x <= y) && any(x < y);  

end
```

## `main.m` 文件代码：

```matlab
% 主程序文件
% 调用MOEA-D算法的主函数

moead;  % 执行MOEA-D算法的实现
```

## `moead.m` 文件代码：

```matlab
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
```

## `MOP2.m` 文件代码：

```matlab
% MOP2多目标优化函数
% 输入参数:
% x - 决策变量的向量

function z = MOP2(x)

    n = numel(x);  % 获取决策变量的数量
    
    % 计算两个目标函数值
    z = [
        1 - exp(-sum((x - 1/sqrt(n)).^2));  % 第一个目标函数
        1 - exp(-sum((x + 1/sqrt(n)).^2))   % 第二个目标函数
    ];
    
end
```

## `PlotCosts.m` 文件代码：

```matlab
% 绘制目标函数代价的函数
% 输入参数:
% EP - 包含非支配个体的数组

function PlotCosts(EP)

    EPC = [EP.Cost];  % 提取所有非支配个体的代价值
    plot(EPC(1, :), EPC(2, :), 'x');  % 绘制目标函数的散点图
    xlabel('1^{st} Objective');  % 设置x轴标签
    ylabel('2^{nd} Objective');  % 设置y轴标签
    grid on;  % 显示网格
    
end
```

## `ZDT.m` 文件代码：

```matlab
% ZDT多目标优化函数
% 输入参数:
% x - 决策变量的向量

function z = ZDT(x)

    n = numel(x);  % 获取决策变量的数量
    f1 = x(1);  % 第一个目标函数值为决策变量的第一个元素
    g = 1 + 9 / (n - 1) * sum(x(2:end));  % 计算g函数，涉及其他决策变量
    h = 1 - sqrt(f1 / g);  % 计算h函数
    f2 = g * h;  % 计算第二个目标函数值
    
    z = [f1;   % 返回目标函数值
         f2];
end
```
