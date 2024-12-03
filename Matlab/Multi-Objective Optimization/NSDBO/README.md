# 非支配排序的蜣螂算法（NSDBO）概述

**非支配排序的蜣螂算法（Non-Dominated Sorting Dung Beetle Optimization, NSDBO）**是一种基于自然启发的多目标优化算法，融合了蜣螂（Dung Beetle）的行为特性和非支配排序技术。该算法旨在解决具有多个冲突目标的优化问题，通过模拟蜣螂在自然界中的导航、寻找食物和构建巢穴的行为，实现高效的搜索和优化过程。

### 1. 蜣螂优化算法（DBO）简介

蜣螂优化算法（Dung Beetle Optimization, DBO）是一种模仿蜣螂在自然界中搜寻食物和构建巢穴行为的群体智能优化算法。蜣螂在移动和搜寻过程中展现出高效的路径规划和资源利用能力，这些特性被转化为优化算法的搜索策略。

**主要特性包括：**
- **导航行为**：蜣螂通过环境中的线索（如气味、声音等）进行导航，寻找食物来源。
- **构建巢穴**：蜣螂在找到食物后，会构建巢穴，用于繁殖和储存食物。
- **群体协作**：多只蜣螂协同工作，分享信息，提高搜索效率。

### 2. 非支配排序的引入

在多目标优化问题中，目标函数往往是相互冲突的，无法同时优化所有目标。为了解决这一问题，引入了**非支配排序（Non-Dominated Sorting）**技术，用于识别和维护一组称为“帕累托前沿（Pareto Front）”的非支配解集。

**非支配排序的核心概念：**
- **支配关系**：解A支配解B，如果A在所有目标上都不劣于B，且至少在一个目标上优于B。
- **帕累托前沿**：由一组相互不支配的解组成，表示在当前搜索空间内的最优解集。

### 3. 非支配排序的蜣螂算法（NSDBO）的工作机制

结合蜣螂优化算法和非支配排序，NSDBO通过以下步骤实现多目标优化：

#### 3.1 初始化阶段
- **种群初始化**：在决策变量的搜索空间内随机生成一组蜣螂个体（解）。
- **非支配排序**：对初始种群进行非支配排序，识别出不同等级的非支配解集。

#### 3.2 适应度评估
- **目标函数计算**：对于每个蜣螂个体，计算其在各个目标函数上的值。
- **非支配排序**：根据计算结果，对种群进行非支配排序，分配不同的等级。

#### 3.3 选择和更新
- **等级选择**：优先选择更高等级（更靠近帕累托前沿）的蜣螂个体进行繁殖和搜索。
- **搜索策略**：模拟蜣螂的导航行为，通过调整个体的位置（决策变量）在搜索空间中进行探索和开发。
  - **局部搜索**：在已知优良区域附近进行细致搜索，提升解的精度。
  - **全局搜索**：在整个搜索空间内进行广泛搜索，避免陷入局部最优。

#### 3.4 环境适应和多样性维护
- **拥挤度计算**：利用非支配排序中的拥挤度信息，保持解集的多样性，防止过度集中在某一部分区域。
- **更新种群**：根据选择和搜索策略，更新种群中的蜣螂个体，保留非支配的优良解。

#### 3.5 终止条件
- **迭代次数**：达到预设的最大迭代次数。
- **收敛标准**：解集在帕累托前沿上的分布趋于稳定。

### 4. NSDBO在多目标测试函数中的应用

结合您提供的 `testmop.m` 文件中定义的多目标测试函数（如 ZDT、DTLZ、WFG、CEC2009 UF 和 CF 系列），NSDBO 可以通过以下方式应用于这些测试问题：

#### 4.1 测试函数的选择
- **无约束问题**：如 ZDT、DTLZ、WFG 和 CEC2009 UF 系列，NSDBO 直接在定义的决策变量范围内搜索，利用非支配排序优化多个目标。
- **有约束问题**：如 CEC2009 CF 系列，NSDBO 在搜索过程中不仅考虑目标函数，还需处理约束条件，通过约束处理机制（如罚函数）维持解的可行性。

#### 4.2 适应度评估
- **目标函数计算**：调用 `p.func` 函数计算每个蜣螂个体在各个目标上的值。
- **非支配排序**：基于目标函数值，对种群进行非支配排序，划分不同等级。

#### 4.3 搜索策略的设计
- **变换函数的利用**：在 WFG 系列问题中，变换函数（如 `s_linear`, `b_flat`, `r_sum` 等）被用于处理输入决策变量。NSDBO 可以结合这些变换函数，增强搜索策略的多样性和适应性。
- **适应性调整**：根据当前的非支配排序结果，动态调整搜索策略的参数，如步长、搜索范围等，提高算法的搜索效率和解的质量。

#### 4.4 多样性维护
- **拥挤度信息**：在非支配排序中计算拥挤度，用于保持解集的多样性，确保帕累托前沿上的解均匀分布。
- **拥挤距离排序**：类似于 NSGA-II，NSDBO 可以结合拥挤距离信息，对解集进行二次排序，优先选择拥挤度较大的解，防止解集过于集中。

### 5. NSDBO的优势与挑战

#### 5.1 优势
- **高效的多目标搜索能力**：结合蜣螂优化算法的群体协作和非支配排序的精确度，NSDBO 能够高效地在多目标空间中搜索优良解。
- **良好的多样性维护**：通过非支配排序和拥挤度信息，NSDBO 能够保持解集的多样性，覆盖整个帕累托前沿。
- **灵活的适应性**：NSDBO 可以通过调整搜索策略和变换函数，适应不同类型的多目标优化问题。

#### 5.2 挑战
- **参数调优**：NSDBO 需要合理设置各种参数（如种群大小、迭代次数、搜索步长等），以确保算法的性能。
- **计算复杂度**：非支配排序和拥挤度计算可能增加算法的计算负担，尤其在高维目标空间和大规模种群时。
- **局部搜索能力**：如何平衡全局搜索和局部搜索，避免陷入局部最优，是提高 NSDBO 性能的关键。

### 6. 结合代码的具体实现示例

假设我们希望使用 NSDBO 解决 ZDT1 问题，以下是一个简化的实现示例，结合了您提供的 `testmop.m` 文件中的 ZDT1 测试函数。

```matlab
% 初始化参数
global M k l;
M = 2;      % 目标函数数量
k = 5;      % DTLZ和WFG等问题的参数，ZDT问题不使用
l = 0;      % DTLZ和WFG等问题的参数，ZDT问题不使用

% 生成测试问题
mop = testmop('zdt1', 30);  % 例如，30维度

% NSDBO参数设置
population_size = 100;
max_iterations = 250;
population = initialize_population(population_size, mop.domain);

% 迭代优化过程
for iter = 1:max_iterations
    % 评价适应度
    fitness = evaluate_fitness(population, mop);
    
    % 非支配排序
    [fronts, crowding_distances] = non_dominated_sort(fitness);
    
    % 选择操作（基于等级和拥挤度）
    selected = selection(population, fronts, crowding_distances, population_size);
    
    % 生成新种群（交叉和变异）
    offspring = generate_offspring(selected, mop);
    
    % 更新种群
    population = offspring;
end

% 提取帕累托前沿
[fronts, ~] = non_dominated_sort(fitness);
pareto_front = fitness(fronts{1}, :);

% 绘制结果
figure;
plot(pareto_front(:,1), pareto_front(:,2), 'ro');
xlabel('Objective 1');
ylabel('Objective 2');
title('Pareto Front Obtained by NSDBO on ZDT1');
```

**注：** 上述代码仅为示意，实际实现需要定义以下函数：
- `initialize_population`：根据决策变量范围随机初始化种群。
- `evaluate_fitness`：调用 `mop.func` 计算种群中每个个体的目标函数值。
- `non_dominated_sort`：实现非支配排序和拥挤度计算。
- `selection`：基于非支配排序和拥挤度信息选择优良个体。
- `generate_offspring`：通过交叉和变异操作生成新一代个体。

### 7. 结论

**非支配排序的蜣螂算法（NSDBO）**通过结合蜣螂优化算法的自然启发搜索策略和非支配排序的多目标优化能力，提供了一种高效、灵活的多目标优化工具。它能够有效地处理具有多个冲突目标的复杂优化问题，保持解集的多样性，并在广泛的测试函数（如ZDT、DTLZ、WFG、CEC2009 UF 和 CF系列）上展现出良好的性能。

然而，NSDBO的实际效果依赖于算法参数的合理设置和变换函数的有效应用。未来的研究可以进一步优化其搜索策略，提升其在高维和复杂约束环境下的适应能力。


---

### `GD.m` 文件

```matlab
% GD.m
% 计算种群目标解与真实Pareto前沿之间的广义差距（Generational Distance）
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
%   PF     - 真实Pareto前沿的目标函数值矩阵
% 输出:
%   Score  - 广义差距值

function Score = GD(PopObj, PF)
    % 计算种群中每个个体到真实Pareto前沿的最小欧几里得距离
    Distance = min(pdist2(PopObj, PF), [], 2);
    
    % 计算所有最小距离的欧几里得范数，并除以种群大小以得到广义差距
    Score    = norm(Distance) / length(Distance);
end
```

**注释说明：**
- **函数功能**：计算种群目标解与真实Pareto前沿之间的广义差距（Generational Distance，GD）。GD用于评估种群解的收敛性，值越小表示种群解越接近真实Pareto前沿。
- **参数说明**：
  - `PopObj`：种群中所有个体的目标函数值，每一行代表一个个体的多个目标值。
  - `PF`：真实Pareto前沿的目标函数值，每一行代表一个Pareto最优解的多个目标值。
- **计算步骤**：
  1. 使用 `pdist2` 函数计算种群中每个个体与真实Pareto前沿之间的欧几里得距离，得到一个距离矩阵。
  2. 对于每个个体，取其与Pareto前沿中所有解的最小距离，得到一个最小距离向量 `Distance`。
  3. 计算所有最小距离的欧几里得范数，并除以种群大小，得到广义差距 `Score`。

---

### `GetFuninfo.m` 文件

```matlab
% GetFuninfo.m
% 获取多目标优化测试函数的信息，包括变量维数、目标数、变量范围、成本函数等
% 输入:
%   TestProblem - 测试问题编号（1到47）
% 输出:
%   MultiObj - 结构体，包含测试问题的各种信息

function MultiObj = GetFunInfo(TestProblem) % 46个多目标测试函数
    dynamic = 0;  % 是否为动态优化问题，0表示静态

    switch TestProblem
        %% 静态多目标 46个
        case 1
            % ZDT1 测试函数
            nVar = 10;                  % 变量维数
            numOfObj = 2;               % 目标数
            mop = testmop('zdt1', nVar);% 初始化测试函数
            CostFunction = @(x) mop.func(x); % 成本函数句柄
            VarMin = mop.domain(:,1)'; % 变量下界
            VarMax = mop.domain(:,2)'; % 变量上界
            name = 'zdt1';              % 测试函数名称
            load('./ParetoFront/ZDT1.mat'); % 加载真实Pareto前沿
            MultiObj.truePF = PF;       % 真实Pareto前沿
        case 2
            % ZDT2 测试函数
            nVar = 10;
            mop = testmop('zdt2', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'zdt2';
            load('./ParetoFront/ZDT2.mat');
            MultiObj.truePF = PF;
        case 3
            % ZDT3 测试函数
            nVar = 10;
            mop = testmop('zdt3', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'zdt3';
            load('./ParetoFront/ZDT3.mat');
            MultiObj.truePF = PF;
        case 4
            % ZDT4 测试函数
            nVar = 10;
            mop = testmop('zdt4', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'zdt4';
            load('./ParetoFront/ZDT4.mat');
            MultiObj.truePF = PF;
        case 5
            % ZDT6 测试函数
            nVar = 10;
            mop = testmop('zdt6', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'zdt6';
            load('./ParetoFront/ZDT6.mat');
            MultiObj.truePF = PF;
        case 6
            % DTLZ1 测试函数
            global k M;
            k = 5;                      % DTLZ1特定参数
            M = 3;                      % 目标数
            nVar = 7;                   % 变量维数
            mop = testmop('DTLZ1', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 3;
            dynamic = 0;
            name = 'DTLZ1';
            load('./ParetoFront/DTLZ1.mat');
            MultiObj.truePF = PF;
        case 7
            % DTLZ2 测试函数
            global k M;
            k = 10;
            M = 3;
            nVar = 12;
            mop = testmop('DTLZ2', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 3;
            dynamic = 0;
            name = 'DTLZ2';
            load('./ParetoFront/DTLZ2.mat');
            MultiObj.truePF = PF;
        case 8
            % DTLZ3 测试函数
            global k M;
            k = 10;
            M = 3;
            nVar = 12;
            mop = testmop('DTLZ3', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 3;
            dynamic = 0;
            name = 'DTLZ3';
            load('./ParetoFront/DTLZ3.mat');
            MultiObj.truePF = PF;
        case 9
            % DTLZ4 测试函数
            global k M;
            k = 10;
            M = 3;
            nVar = 12;
            mop = testmop('DTLZ4', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 3;
            dynamic = 0;
            name = 'DTLZ4';
            load('./ParetoFront/DTLZ4.mat');
            MultiObj.truePF = PF;
        case 10
            % DTLZ5 测试函数
            global k M;
            k = 10;
            M = 3;
            nVar = 12;
            mop = testmop('DTLZ5', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 3;
            dynamic = 0;
            name = 'DTLZ5';
            load('./ParetoFront/DTLZ5.mat');
            MultiObj.truePF = PF;
        case 11
            % DTLZ6 测试函数
            global k M;
            k = 10;
            M = 3;
            nVar = 12;
            mop = testmop('DTLZ6', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 3;
            dynamic = 0;
            name = 'DTLZ6';
            load('./ParetoFront/DTLZ6.mat');
            MultiObj.truePF = PF;
        case 12
            % DTLZ7 测试函数
            global k M;
            k = 20;
            M = 3;
            nVar = 22;
            mop = testmop('DTLZ7', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 3;
            name = 'DTLZ7';
            load('./ParetoFront/DTLZ7.mat');
            MultiObj.truePF = PF;
        case 13
            % WFG1 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg1', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg1';
            load('./ParetoFront/wfg1.mat');
            MultiObj.truePF = PF;
        case 14
            % WFG2 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg2', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg2';
            load('./ParetoFront/wfg2.mat');
            MultiObj.truePF = PF;
        case 15
            % WFG3 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg3', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg3';
            load('./ParetoFront/wfg3.mat');
            MultiObj.truePF = PF;
        case 16
            % WFG4 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg4', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg4';
            load('./ParetoFront/wfg4.mat');
            MultiObj.truePF = PF;
        case 17
            % WFG5 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg5', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg5';
            load('./ParetoFront/wfg5.mat');
            MultiObj.truePF = PF;
        case 18
            % WFG6 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg6', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg6';
            load('./ParetoFront/wfg6.mat');
            MultiObj.truePF = PF;
        case 19
            % WFG7 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg7', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg7';
            load('./ParetoFront/wfg7.mat');
            MultiObj.truePF = PF;
        case 20
            % WFG8 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg8', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg8';
            load('./ParetoFront/wfg8.mat');
            MultiObj.truePF = PF;
        case 21
            % WFG9 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg9', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg9';
            load('./ParetoFront/wfg9.mat');
            MultiObj.truePF = PF;
        case 22
            % WFG10 测试函数
            global k l M;
            k = 2;
            l = 4;
            M = 2;
            nVar = 6;
            mop = testmop('wfg10', nVar);
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            numOfObj = 2;
            name = 'wfg10';
            load('./ParetoFront/wfg10.mat');
            MultiObj.truePF = PF;
        case 23
            % UF1 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('uf1', nVar); % 初始化UF1测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf1';
            load('./ParetoFront/UF1.mat');
            MultiObj.truePF = PF;
        case 24
            % UF2 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('uf2', nVar); % 初始化UF2测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf2';
            load('./ParetoFront/UF2.mat');
            MultiObj.truePF = PF;
        case 25
            % UF3 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('uf3', nVar); % 初始化UF3测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf3';
            load('./ParetoFront/UF3.mat');
            MultiObj.truePF = PF;
        case 26
            % UF4 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('uf4', nVar); % 初始化UF4测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf4';
            load('./ParetoFront/UF4.mat');
            MultiObj.truePF = PF;
        case 27
            % UF5 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('uf5', nVar); % 初始化UF5测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf5';
            load('./ParetoFront/UF5.mat');
            MultiObj.truePF = PF;
        case 28
            % UF6 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('uf6', nVar); % 初始化UF6测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf6';
            load('./ParetoFront/UF6.mat');
            MultiObj.truePF = PF;
        case 29
            % UF7 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('uf7', nVar); % 初始化UF7测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf7';
            load('./ParetoFront/UF7.mat');
            MultiObj.truePF = PF;
        case 30
            % UF8 测试函数
            nVar = 10;
            numOfObj = 3;
            mop = testmop('uf8', nVar); % 初始化UF8测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf8';
            load('./ParetoFront/UF8.mat');
            MultiObj.truePF = PF;
        case 31
            % UF9 测试函数
            nVar = 10;
            numOfObj = 3;
            mop = testmop('uf9', nVar); % 初始化UF9测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf9';
            load('./ParetoFront/UF9.mat');
            MultiObj.truePF = PF;
        case 32
            % UF10 测试函数
            nVar = 10;
            numOfObj = 3;
            mop = testmop('uf10', nVar); % 初始化UF10测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'uf10';
            load('./ParetoFront/UF10.mat');
            MultiObj.truePF = PF;
        case 33
            % CF1 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('cf1', nVar); % 初始化CF1测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf1';
            load('./ParetoFront/CF1.mat');
            MultiObj.truePF = PF;
        case 34
            % CF2 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('cf2', nVar); % 初始化CF2测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf2';
            load('./ParetoFront/CF2.mat');
            MultiObj.truePF = PF;
        case 35
            % CF3 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('cf3', nVar); % 初始化CF3测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf3';
            load('./ParetoFront/CF3.mat');
            MultiObj.truePF = PF;
        case 36
            % CF4 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('cf4', nVar); % 初始化CF4测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf4';
            load('./ParetoFront/CF4.mat');
            MultiObj.truePF = PF;
        case 37
            % CF5 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('cf5', nVar); % 初始化CF5测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf5';
            load('./ParetoFront/CF5.mat');
            MultiObj.truePF = PF;
        case 38
            % CF6 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('cf6', nVar); % 初始化CF6测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf6';
            load('./ParetoFront/CF6.mat');
            MultiObj.truePF = PF;
        case 39
            % CF7 测试函数
            nVar = 10;
            numOfObj = 2;
            mop = testmop('cf7', nVar); % 初始化CF7测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf7';
            load('./ParetoFront/CF7.mat');
            MultiObj.truePF = PF;
        case 40
            % CF8 测试函数
            nVar = 10;
            numOfObj = 3;
            mop = testmop('cf8', nVar); % 初始化CF8测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf8';
            load('./ParetoFront/CF8.mat');
            MultiObj.truePF = PF;
        case 41
            % CF9 测试函数
            nVar = 10;
            numOfObj = 3;
            mop = testmop('cf9', nVar); % 初始化CF9测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf9';
            load('./ParetoFront/CF9.mat');
            MultiObj.truePF = PF;
        case 42
            % CF10 测试函数
            nVar = 10;
            numOfObj = 3;
            mop = testmop('cf10', nVar); % 初始化CF10测试函数
            CostFunction = @(x) mop.func(x);
            VarMin = mop.domain(:,1)';
            VarMax = mop.domain(:,2)';
            name = 'cf10';
            load('./ParetoFront/CF10.mat');
            MultiObj.truePF = PF;
        case 43
            % Kursawe 测试函数
            CostFunction = @(x) [ ...
                -10 .* (exp(-0.2 .* sqrt(x(:,1).^2 + x(:,2).^2)) + exp(-0.2 .* sqrt(x(:,2).^2 + x(:,3).^2))); ...
                sum(abs(x).^0.8 + 5 .* sin(x.^3), 2) ...
            ];
            nVar = 3;                        % 变量维数
            VarMin = -5 .* ones(1, nVar);    % 变量下界
            VarMax = 5 .* ones(1, nVar);     % 变量上界
            load('./ParetoFront/Kursawe.mat');% 加载真实Pareto前沿
            MultiObj.truePF = PF;
            name = 'Kursawe';                % 测试函数名称
            numOfObj = 2;                     % 目标数
        case 44
            % Poloni's 两目标测试函数
            A1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2);
            A2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2);
            B1 = @(x, y) 0.5 .* sin(x) - 2 .* cos(x) + sin(y) - 1.5 .* cos(y);
            B2 = @(x, y) 1.5 .* sin(x) - cos(x) + 2 .* sin(y) - 0.5 .* cos(y);
            f1 = @(x, y) 1 + (A1 - B1(x, y)).^2 + (A2 - B2(x, y)).^2;
            f2 = @(x, y) (x + 3).^2 + (y + 1).^2;
            CostFunction = @(x) [f1(x(:,1), x(:,2)); f2(x(:,1), x(:,2))];
            nVar = 2;                              % 变量维数
            VarMin = -pi .* ones(1, nVar);         % 变量下界
            VarMax = pi .* ones(1, nVar);          % 变量上界
            name = 'Poloni';                       % 测试函数名称
            numOfObj = 2;                           % 目标数
        case 45
            % Viennet2 三目标测试函数
            f1 = @(x, y) 0.5 .* (x - 2).^2 + (1/13) .* (y + 1).^2 + 3;
            f2 = @(x, y) (1/36) .* (x + y - 3).^2 + (1/8) .* (-x + y + 2).^2 - 17;
            f3 = @(x, y) (1/175) .* (x + 2 .* y - 1).^2 + (1/17) .* (2 .* y - x).^2 - 13;
            CostFunction = @(x) [f1(x(:,1), x(:,2)); f2(x(:,1), x(:,2)); f3(x(:,1), x(:,2))];
            nVar = 2;                            % 变量维数
            VarMin = [-4, -4];                   % 变量下界
            VarMax = [4, 4];                     % 变量上界
            load('./ParetoFront/Viennet2.mat');  % 加载真实Pareto前沿
            MultiObj.truePF = PF;
            name = 'Viennet2';                   % 测试函数名称
            numOfObj = 3;                        % 目标数
        case 46
            % Viennet3 三目标测试函数
            f1 = @(x, y) 0.5 .* (x.^2 + y.^2) + sin(x.^2 + y.^2);
            f2 = @(x, y) (1/8) .* (3 .* x - 2 .* y + 4).^2 + (1/27) .* (x - y + 1).^2 + 15;
            f3 = @(x, y) (1 ./ (x.^2 + y.^2 + 1)) - 1.1 .* exp(-(x.^2 + y.^2));
            CostFunction = @(x) [f1(x(:,1), x(:,2)); f2(x(:,1), x(:,2)); f3(x(:,1), x(:,2))];
            nVar = 2;                            % 变量维数
            VarMin = [-3, -10];                  % 变量下界
            VarMax = [10, 3];                    % 变量上界
            load('./ParetoFront/Viennet3.mat');  % 加载真实Pareto前沿
            MultiObj.truePF = PF;
            name = 'Viennet3';                   % 测试函数名称
            numOfObj = 3;                        % 目标数
        case 47
            % 盘式制动器设计问题（自定义问题）
            f1 = @(r, R, F, s) 4.9 .* 10.^(-5) .* (R.^2 - r.^2) .* (s - 1);
            f2 = @(r, R, F, s) 9.82 .* 10.^6 .* (R.^2 - r.^2) ./ (F .* s .* (R.^3 - r.^3));
            % 约束条件
            g1 = @(r, R, F, s) 20 - (R - r);
            g2 = @(r, R, F, s) 2.5 .* (s + 1) - 30;
            g3 = @(r, R, F, s) F ./ (3.14 .* (R.^2 - r.^2)) - 0.4;
            g4 = @(r, R, F, s) 2.22 .* 10.^(-3) .* F .* (R.^3 - r.^3) ./ (R.^2 - r.^2).^2 - 1;
            g5 = @(r, R, F, s) 900 - 0.0266 .* F .* s .* (R.^3 - r.^3) ./ (R.^2 - r.^2);
            % 总约束函数
            g = @(r, R, F, s) 10.^3 .* (max(0, g1(r, R, F, s)) + max(0, g2(r, R, F, s)) + ...
                max(0, g3(r, R, F, s)) + max(0, g4(r, R, F, s)) + max(0, g5(r, R, F, s)));
            % 成本函数，包含目标函数和约束惩罚
            CostFunction = @(x) [ ...
                f1(x(:,1), x(:,2), x(:,3), x(:,4)) + g(x(:,1), x(:,2), x(:,3), x(:,4)); ...
                f2(x(:,1), x(:,2), x(:,3), x(:,4)) + g(x(:,1), x(:,2), x(:,3), x(:,4)) ...
            ];
            nVar = 4;                             % 变量维数
            VarMin = [55, 75, 1000, 2];           % 变量下界
            VarMax = [80, 110, 3000, 20];         % 变量上界
            name = '盘式制动器设计';                % 测试函数名称
            numOfObj = 2;                         % 目标数
    end

    % 将测试问题的信息封装到 MultiObj 结构体中
    MultiObj.nVar     = nVar;        % 变量维数
    MultiObj.var_min  = VarMin;      % 变量下界
    MultiObj.var_max  = VarMax;      % 变量上界
    MultiObj.fun      = CostFunction;% 成本函数句柄
    MultiObj.dynamic  = dynamic;     % 是否为动态优化问题
    MultiObj.numOfObj = numOfObj;    % 目标数
    MultiObj.name     = name;        % 测试函数名称
end
```

**注释说明：**
- **函数功能**：`GetFunInfo` 函数根据输入的测试问题编号，返回相应多目标优化问题的详细信息，包括变量维数、目标数、变量范围、成本函数、真实Pareto前沿等。
- **参数说明**：
  - `TestProblem`：测试问题的编号，范围为1到47，对应不同的多目标测试函数。
- **输出说明**：
  - `MultiObj`：一个结构体，包含以下字段：
    - `nVar`：变量维数。
    - `var_min`：变量的下界。
    - `var_max`：变量的上界。
    - `fun`：成本函数句柄，用于计算目标函数值。
    - `dynamic`：是否为动态优化问题，0表示静态。
    - `numOfObj`：目标数。
    - `name`：测试函数的名称。
    - `truePF`：真实Pareto前沿的目标函数值，用于评估算法性能。

- **代码结构**：
  - 使用 `switch` 语句根据 `TestProblem` 的值选择不同的测试函数。
  - 对于每个测试函数，设置相应的变量维数、目标数、变量范围、成本函数，并加载对应的真实Pareto前沿数据。
  - 特别注意：
    - **测试函数类型**：
      - ZDT系列（ZDT1到ZDT6）：常用的双目标测试函数。
      - DTLZ系列（DTLZ1到DTLZ7）：适用于更多目标数的测试函数。
      - WFG系列（wfg1到wfg10）：具有不同特性的多目标测试函数。
      - UF系列（uf1到uf10）：双目标和三目标测试函数。
      - CF系列（cf1到cf10）：带有约束的多目标测试函数。
      - 其他自定义问题（如Kursawe、Viennet2、Viennet3、盘式制动器设计等）。
    - **成本函数**：根据测试函数的定义，构建相应的成本函数句柄 `CostFunction`，用于计算个体的目标函数值。
    - **真实Pareto前沿**：通过 `load` 函数加载预先计算好的真实Pareto前沿数据，存储在 `MultiObj.truePF` 中，供算法性能评估使用。
    - **约束处理**：对于带有约束的测试函数（如盘式制动器设计），在成本函数中加入惩罚项 `g`，以处理约束条件。

- **注意事项**：
  - 确保在指定路径下存在相应的 `.mat` 文件，这些文件包含了预先计算好的真实Pareto前沿数据。
  - 对于自定义测试函数（如案例47），需要根据具体问题定义目标函数和约束条件，并正确设置变量范围。

---

### `HV.m` 文件

```matlab
% HV.m
% 计算种群的超体积（Hypervolume）指标
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
%   PF     - 真实Pareto前沿的目标函数值矩阵
% 输出:
%   Score  - 种群的超体积值
%   PopObj - 经过归一化处理后的种群目标函数值矩阵

function [Score, PopObj] = HV(PopObj, PF)

    % 获取种群和Pareto前沿的维数
    [N, M] = size(PopObj);  % N为种群个体数，M为目标维数

    % 计算种群的最小值，取每个目标的最小值和0中的较小值
    fmin = min(min(PopObj, [], 1), zeros(1, M));

    % 计算Pareto前沿的最大值，作为归一化的上界
    fmax = max(PF, [], 1);

    % 对种群进行归一化处理
    % 归一化公式: (x - fmin) / ((fmax - fmin) * 1.1)
    PopObj = (PopObj - repmat(fmin, N, 1)) ./ repmat((fmax - fmin) * 1.1, N, 1);

    % 移除任何在归一化后超过1的个体
    PopObj(any(PopObj > 1, 2), :) = [];

    % 设置参考点为所有目标维度的1
    RefPoint = ones(1, M);

    if isempty(PopObj)
        % 如果种群为空，则超体积为0
        Score = 0;
    elseif M < 4
        % 当目标维数小于4时，计算精确的超体积值

        % 按照所有目标从小到大排序种群
        pl = sortrows(PopObj);

        % 初始化S集合，包含权重和排序后的点
        S = {1, pl};

        % 对每个维度进行切片
        for k = 1 : M-1
            S_ = {};  % 临时存储新的S集合

            for i = 1 : size(S, 1)
                % 对当前维度进行切片
                Stemp = Slice(cell2mat(S(i, 2)), k, RefPoint);

                for j = 1 : size(Stemp, 1)
                    % 更新权重和切片后的点
                    temp(1) = {cell2mat(Stemp(j, 1)) * cell2mat(S(i, 1))};
                    temp(2) = Stemp(j, 2);
                    % 将新的切片结果加入临时集合
                    S_ = Add(temp, S_);
                end
            end

            % 更新S集合
            S = S_;
        end

        % 初始化超体积得分
        Score = 0;

        for i = 1 : size(S, 1)
            % 获取当前切片的头部点
            p = Head(cell2mat(S(i, 2)));
            % 计算超体积贡献
            Score = Score + cell2mat(S(i, 1)) * abs(p(M) - RefPoint(M));
        end
    else
        % 当目标维数大于等于4时，使用蒙特卡洛方法估计超体积值

        % 设置采样点数量
        SampleNum = 1000000;

        % 设置最大值和最小值用于采样
        MaxValue = RefPoint;
        MinValue = min(PopObj, [], 1);

        % 生成均匀分布的采样点
        Samples = unifrnd(repmat(MinValue, SampleNum, 1), repmat(MaxValue, SampleNum, 1));

        if gpuDeviceCount > 0
            % 如果有GPU设备可用，使用GPU加速
            Samples = gpuArray(single(Samples));
            PopObj  = gpuArray(single(PopObj));
        end

        for i = 1 : size(PopObj, 1)
            drawnow();  % 更新图形窗口，防止MATLAB界面冻结

            % 初始化支配标记，所有采样点初始为被支配
            domi = true(size(Samples, 1), 1);
            m = 1;

            while m <= M && any(domi)
                % 检查每个目标维度是否被当前个体支配
                domi = domi & PopObj(i, m) <= Samples(:, m);
                m = m + 1;
            end

            % 移除被当前个体支配的采样点
            Samples(domi, :) = [];
        end

        % 计算超体积得分
        Score = prod(MaxValue - MinValue) * (1 - size(Samples, 1) / SampleNum);
    end
end

% Slice 函数
% 对给定点集进行切片操作
% 输入:
%   pl        - 点集
%   k         - 当前维度
%   RefPoint  - 参考点
% 输出:
%   S         - 切片结果

function S = Slice(pl, k, RefPoint)
    p  = Head(pl);    % 获取点集的第一个点
    pl = Tail(pl);    % 获取点集的剩余部分
    ql = [];          % 初始化切片后的点集
    S  = {};          % 初始化切片结果集合

    while ~isempty(pl)
        ql  = Insert(p, k+1, ql);  % 插入点到切片集合
        p_  = Head(pl);            % 获取下一个点
        cell_(1,1) = {abs(p(k) - p_(k))};  % 计算当前维度的差值
        cell_(1,2) = {ql};                  % 关联切片后的点集
        S   = Add(cell_, S);                % 添加到切片结果集合
        p   = p_;                            % 更新当前点
        pl  = Tail(pl);                      % 更新点集
    end

    ql = Insert(p, k+1, ql);  % 插入最后一个点
    cell_(1,1) = {abs(p(k) - RefPoint(k))};  % 计算与参考点的差值
    cell_(1,2) = {ql};                        % 关联切片后的点集
    S  = Add(cell_, S);                      % 添加到切片结果集合
end

% Insert 函数
% 将点插入到切片集合中，并处理点的顺序和支配关系
% 输入:
%   p  - 当前点
%   k  - 当前维度
%   pl - 切片后的点集
% 输出:
%   ql - 更新后的切片点集

function ql = Insert(p, k, pl)
    flag1 = 0;
    flag2 = 0;
    ql    = [];
    hp    = Head(pl);  % 获取切片点集的第一个点

    % 将小于当前点的点加入到切片点集中
    while ~isempty(pl) && hp(k) < p(k)
        ql = [ql; hp];
        pl = Tail(pl);
        hp = Head(pl);
    end

    ql = [ql; p];  % 插入当前点

    m  = length(p);  % 点的维数

    while ~isempty(pl)
        q = Head(pl);  % 获取下一个点
        for i = k : m
            if p(i) < q(i)
                flag1 = 1;
            else
                if p(i) > q(i)
                    flag2 = 1;
                end
            end
        end

        % 如果当前点p不完全支配点q，则保留点q
        if ~(flag1 == 1 && flag2 == 0)
            ql = [ql; Head(pl)];
        end

        pl = Tail(pl);  % 更新切片点集
    end
end

% Head 函数
% 获取点集的第一个点
% 输入:
%   pl - 点集
% 输出:
%   p  - 第一个点

function p = Head(pl)
    if isempty(pl)
        p = [];
    else
        p = pl(1, :);  % 返回第一个点
    end
end

% Tail 函数
% 获取点集的剩余部分（去除第一个点）
% 输入:
%   pl - 点集
% 输出:
%   ql - 剩余点集

function ql = Tail(pl)
    if size(pl, 1) < 2
        ql = [];
    else
        ql = pl(2:end, :);  % 返回除第一个点外的所有点
    end
end

% Add 函数
% 将切片结果添加到切片集合中，处理权重的累加
% 输入:
%   cell_ - 当前切片的权重和点集
%   S     - 现有的切片集合
% 输出:
%   S_    - 更新后的切片集合

function S_ = Add(cell_, S)
    n = size(S, 1);  % 当前切片集合的大小
    m = 0;
    for k = 1 : n
        if isequal(cell_(1,2), S(k,2))
            % 如果当前点集已存在于切片集合中，则累加权重
            S(k,1) = {cell2mat(S(k,1)) + cell2mat(cell_(1,1))};
            m = 1;
            break;
        end
    end
    if m == 0
        % 如果当前点集不存在于切片集合中，则添加新的切片
        S(n+1, :) = cell_(1, :);
    end
    S_ = S;  % 返回更新后的切片集合
end
```

---

**注释说明：**

- **函数功能概述：**
  - `HV` 函数用于计算给定种群的超体积（Hypervolume）指标。超体积是一个常用的多目标优化评价指标，衡量种群覆盖真实Pareto前沿的程度。其值越大，表示种群在目标空间中覆盖的范围越广。
  
- **主要步骤：**
  1. **归一化处理：**
     - 首先，将种群的目标函数值进行归一化，以便在统一的尺度下计算超体积。归一化的下界为每个目标的最小值和0中的较小值，上界为真实Pareto前沿的最大值的1.1倍。
     - 归一化后的种群中，任何目标值超过1的个体都会被移除。
  
  2. **超体积计算：**
     - 当目标维数小于4时，使用精确的方法计算超体积。这包括对种群点集进行排序、切片和累加权重。
     - 当目标维数大于等于4时，使用蒙特卡洛方法进行超体积估计。通过大量随机采样点，计算被种群支配的采样点比例，从而估计超体积。

- **辅助函数说明：**
  - `Slice`：对给定点集进行切片操作，分割目标空间以计算超体积。
  - `Insert`：将一个点插入到切片点集中，并处理点的顺序和支配关系。
  - `Head`：获取点集的第一个点。
  - `Tail`：获取点集的剩余部分（去除第一个点）。
  - `Add`：将切片结果添加到切片集合中，处理权重的累加。

- **GPU加速：**
  - 当系统中有可用的GPU设备时，蒙特卡洛估计方法会利用GPU进行加速计算，以提高计算效率。

- **注意事项：**
  - **种群为空的处理：** 如果经过归一化后种群为空（即所有个体在至少一个目标上超出归一化范围），则超体积得分设为0。
  - **超体积精确计算的适用性：** 精确计算方法适用于目标维数较低（M < 4）的情况，随着目标维数的增加，计算复杂度急剧增加，因此对于高维目标空间，使用蒙特卡洛估计更为实际。
  - **参考点的选择：** 参考点设置为所有目标维度的1，确保所有种群点均在参考点的支配范围内。

---

**具体代码注释：**

- **归一化处理部分：**

  ```matlab
  [N, M] = size(PopObj);  % 获取种群的个体数和目标维数
  fmin = min(min(PopObj, [], 1), zeros(1, M));  % 计算每个目标的最小值和0中的较小值
  fmax = max(PF, [], 1);  % 计算Pareto前沿中每个目标的最大值
  PopObj = (PopObj - repmat(fmin, N, 1)) ./ repmat((fmax - fmin) * 1.1, N, 1);  % 归一化种群
  PopObj(any(PopObj > 1, 2), :) = [];  % 移除任何在归一化后超过1的个体
  RefPoint = ones(1, M);  % 设置参考点为所有目标维度的1
  ```

- **超体积计算部分：**

  ```matlab
  if isempty(PopObj)
      Score = 0;  % 如果种群为空，超体积为0
  elseif M < 4
      % 精确计算超体积
      pl = sortrows(PopObj);  % 按照所有目标从小到大排序种群
      S = {1, pl};  % 初始化切片结果集合
      for k = 1 : M-1
          S_ = {};  % 临时存储新的切片结果
          for i = 1 : size(S, 1)
              Stemp = Slice(cell2mat(S(i, 2)), k, RefPoint);  % 切片操作
              for j = 1 : size(Stemp, 1)
                  temp(1) = {cell2mat(Stemp(j, 1)) * cell2mat(S(i, 1))};  % 更新权重
                  temp(2) = Stemp(j, 2);  % 更新点集
                  S_ = Add(temp, S_);  % 添加到临时切片结果集合
              end
          end
          S = S_;  % 更新切片结果集合
      end
      Score = 0;  % 初始化超体积得分
      for i = 1 : size(S, 1)
          p = Head(cell2mat(S(i, 2)));  % 获取当前切片的头部点
          Score = Score + cell2mat(S(i, 1)) * abs(p(M) - RefPoint(M));  % 计算超体积贡献
      end
  else
      % 蒙特卡洛估计超体积
      SampleNum = 1000000;  % 设置采样点数量
      MaxValue = RefPoint;  % 设置采样上界为参考点
      MinValue = min(PopObj, [], 1);  % 设置采样下界为种群最小值
      Samples = unifrnd(repmat(MinValue, SampleNum, 1), repmat(MaxValue, SampleNum, 1));  % 生成均匀采样点
      if gpuDeviceCount > 0
          % 如果有GPU可用，使用GPU加速
          Samples = gpuArray(single(Samples));
          PopObj  = gpuArray(single(PopObj));
      end
      for i = 1 : size(PopObj, 1)
          drawnow();  % 更新图形窗口，防止MATLAB界面冻结
          domi = true(size(Samples, 1), 1);  % 初始化支配标记
          m = 1;
          while m <= M && any(domi)
              domi = domi & PopObj(i, m) <= Samples(:, m);  % 检查每个目标维度是否被当前个体支配
              m = m + 1;
          end
          Samples(domi, :) = [];  % 移除被当前个体支配的采样点
      end
      Score = prod(MaxValue - MinValue) * (1 - size(Samples, 1) / SampleNum);  % 计算超体积得分
  end
  ```

- **辅助函数部分：**

  - **Slice 函数：**

    ```matlab
    function S = Slice(pl, k, RefPoint)
        p  = Head(pl);    % 获取点集的第一个点
        pl = Tail(pl);    % 获取点集的剩余部分
        ql = [];          % 初始化切片后的点集
        S  = {};          % 初始化切片结果集合

        while ~isempty(pl)
            ql  = Insert(p, k+1, ql);  % 插入点到切片集合
            p_  = Head(pl);            % 获取下一个点
            cell_(1,1) = {abs(p(k) - p_(k))};  % 计算当前维度的差值
            cell_(1,2) = {ql};                  % 关联切片后的点集
            S   = Add(cell_, S);                % 添加到切片结果集合
            p   = p_;                            % 更新当前点
            pl  = Tail(pl);                      % 更新点集
        end

        ql = Insert(p, k+1, ql);  % 插入最后一个点
        cell_(1,1) = {abs(p(k) - RefPoint(k))};  % 计算与参考点的差值
        cell_(1,2) = {ql};                        % 关联切片后的点集
        S  = Add(cell_, S);                      % 添加到切片结果集合
    end
    ```

  - **Insert 函数：**

    ```matlab
    function ql = Insert(p, k, pl)
        flag1 = 0;
        flag2 = 0;
        ql    = [];
        hp    = Head(pl);  % 获取切片点集的第一个点

        % 将小于当前点的点加入到切片点集中
        while ~isempty(pl) && hp(k) < p(k)
            ql = [ql; hp];
            pl = Tail(pl);
            hp = Head(pl);
        end

        ql = [ql; p];  % 插入当前点

        m  = length(p);  % 点的维数

        while ~isempty(pl)
            q = Head(pl);  % 获取下一个点
            for i = k : m
                if p(i) < q(i)
                    flag1 = 1;
                else
                    if p(i) > q(i)
                        flag2 = 1;
                    end
                end
            end

            % 如果当前点p不完全支配点q，则保留点q
            if ~(flag1 == 1 && flag2 == 0)
                ql = [ql; Head(pl)];
            end

            pl = Tail(pl);  % 更新切片点集
        end
    end
    ```

  - **Head 函数：**

    ```matlab
    function p = Head(pl)
        if isempty(pl)
            p = [];
        else
            p = pl(1, :);  % 返回第一个点
        end
    end
    ```

  - **Tail 函数：**

    ```matlab
    function ql = Tail(pl)
        if size(pl, 1) < 2
            ql = [];
        else
            ql = pl(2:end, :);  % 返回除第一个点外的所有点
        end
    end
    ```

  - **Add 函数：**

    ```matlab
    function S_ = Add(cell_, S)
        n = size(S, 1);  % 当前切片集合的大小
        m = 0;
        for k = 1 : n
            if isequal(cell_(1,2), S(k,2))
                % 如果当前点集已存在于切片集合中，则累加权重
                S(k,1) = {cell2mat(S(k,1)) + cell2mat(cell_(1,1))};
                m = 1;
                break;
            end
        end
        if m == 0
            % 如果当前点集不存在于切片集合中，则添加新的切片
            S(n+1, :) = cell_(1, :);
        end
        S_ = S;  % 返回更新后的切片集合
    end
    ```

---

**详细说明：**

1. **归一化处理：**
   - **目的：** 统一不同目标维度的尺度，使其在计算超体积时具有可比性。
   - **方法：**
     - 计算每个目标的最小值（与0的较小值）和最大值（根据Pareto前沿）。
     - 使用线性归一化将种群的目标值缩放到[0, 1.1]区间内。
     - 移除任何在归一化后超过1的个体，确保所有保留的个体都在参考点的支配范围内。

2. **超体积计算：**
   - **精确计算（M < 4）：**
     - 对种群进行多维切片，逐步计算每个切片的超体积贡献。
     - 使用递归和集合操作，精确计算每个维度的覆盖面积。
     - 最终累加所有切片的贡献，得到总的超体积值。
   - **蒙特卡洛估计（M >= 4）：**
     - 随机生成大量采样点，覆盖目标空间。
     - 判断哪些采样点被种群支配（即被种群中的某个个体支配）。
     - 通过支配的采样点比例估计超体积，效率较高，适用于高维目标空间。

3. **辅助函数的作用：**
   - **`Slice`：** 将高维空间切分为多个低维切片，以便逐步计算超体积。
   - **`Insert`：** 将新的点插入到切片集合中，并确保点的顺序和支配关系得到正确处理。
   - **`Head` 和 `Tail`：** 分别获取点集的第一个点和剩余点，便于递归处理。
   - **`Add`：** 将切片的权重和点集累加到切片集合中，确保切片结果的正确性。

4. **GPU加速：**
   - **条件：** 如果系统中存在可用的GPU设备，则将采样点和种群目标值转移到GPU内存中进行计算。
   - **优势：** 利用GPU的并行计算能力，大幅提高蒙特卡洛估计的计算速度，尤其在高采样点数量时效果显著。

5. **函数返回值：**
   - **`Score`：** 计算得到的超体积值，反映种群覆盖目标空间的程度。
   - **`PopObj`：** 经过归一化处理后的种群目标函数值，用于后续可能的分析或绘图。

6. **性能优化：**
   - **避免不必要的计算：** 在蒙特卡洛估计中，通过逐步移除被支配的采样点，减少后续判断的计算量。
   - **并行计算：** 通过GPU加速，尤其在处理大规模采样点时，显著提升计算效率。

---

### `IGD.m` 文件

```matlab
% IGD.m
% 计算种群的反向广义差距（Inverted Generational Distance, IGD）指标
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
%   PF     - 真实Pareto前沿的目标函数值矩阵
% 输出:
%   Score  - 反向广义差距值

function Score = IGD(PopObj, PF)
    % 计算每个真实Pareto前沿点到种群中最近个体的欧几里得距离
    Distance = min(pdist2(PF, PopObj), [], 2);
    
    % 计算所有距离的平均值，得到反向广义差距
    Score    = mean(Distance);
end
```

**注释说明：**
- **函数功能**：`IGD` 函数用于计算反向广义差距（Inverted Generational Distance, IGD）指标。IGD用于评估种群解集相对于真实Pareto前沿的覆盖情况。值越小表示种群解集覆盖真实Pareto前沿的程度越高。
- **参数说明**：
  - `PopObj`：种群中所有个体的目标函数值，每一行代表一个个体的多个目标值。
  - `PF`：真实Pareto前沿的目标函数值，每一行代表一个Pareto最优解的多个目标值。
- **计算步骤**：
  1. 使用 `pdist2` 函数计算真实Pareto前沿中每个点到种群中所有个体的欧几里得距离，得到一个距离矩阵。
  2. 对于每个真实Pareto前沿点，取其与种群中所有个体的最小距离，得到一个最小距离向量 `Distance`。
  3. 计算所有最小距离的平均值，得到反向广义差距 `Score`。

---

### `initialize variables.m` 文件

```matlab
% initialize_variables.m
% 初始化种群的决策变量和目标函数值
% 参考自NSGA-II，版权所有。
% 输入:
%   NP                - 种群规模（Population size）
%   M                 - 目标函数数量（Number of objective functions）
%   D                 - 决策变量数量（Number of decision variables）
%   LB                - 决策变量下界（Lower bounds）
%   UB                - 决策变量上界（Upper bounds）
%   evaluate_objective - 目标函数评估函数句柄
% 输出:
%   f                 - 初始化后的种群矩阵，包含决策变量和目标函数值

function f = initialize_variables(NP, M, D, LB, UB, evaluate_objective)
    
    % 将下界和上界赋值给min和max
    min = LB;
    max = UB;
    
    % K 是数组元素的总数。为了便于计算，决策变量和目标函数被连接成一个单一的数组。
    % 交叉和变异仅使用决策变量，而选择仅使用目标函数。
    K = M + D;
    
    % 初始化种群矩阵，大小为NP x K，初始值为零
    f = zeros(NP, K);
    
    %% 初始化种群中的每个个体
    % 对于每个染色体执行以下操作（N是种群规模）
    for i = 1 : NP
        % 根据决策变量的最小和最大可能值初始化决策变量。
        % 对于每个决策变量，随机生成一个在[min(j), max(j)]区间内的值
        for j = 1 : D
            f(i, j) = min(j) + (max(j) - min(j)) * rand(1);
        end % 结束j循环
        
        % 为了便于计算和数据处理，染色体还在末尾连接了目标函数值。
        % 从D+1到K的元素存储目标函数值。
        % evaluate_objective函数一次评估一个个体的目标函数值，
        % 仅传递决策变量，并返回目标函数值。这些值存储在个体的末尾。
        f(i, D + 1 : K) = evaluate_objective(f(i, 1:D));
    end % 结束i循环
end
```

**注释说明：**
- **函数功能**：`initialize_variables` 函数用于初始化种群的决策变量和目标函数值。决策变量根据给定的上下界随机生成，目标函数值通过调用 `evaluate_objective` 函数计算得到。
- **参数说明**：
  - `NP`：种群规模，即种群中个体的数量。
  - `M`：目标函数的数量。
  - `D`：决策变量的数量。
  - `LB`：决策变量的下界，长度为D的向量。
  - `UB`：决策变量的上界，长度为D的向量。
  - `evaluate_objective`：一个函数句柄，用于评估单个个体的目标函数值。函数接受一个长度为D的决策变量向量，返回一个长度为M的目标函数值向量。
- **输出说明**：
  - `f`：初始化后的种群矩阵，大小为NP x (D + M)。前D列为决策变量，后M列为目标函数值。
- **初始化步骤**：
  1. **决策变量初始化**：
     - 对于种群中的每个个体，针对每个决策变量，生成一个在其对应下界和上界之间的随机值。
  2. **目标函数值计算**：
     - 对于每个个体，使用 `evaluate_objective` 函数计算其目标函数值，并将结果存储在种群矩阵的后M列。
- **注意事项**：
  - 确保 `LB` 和 `UB` 的长度与决策变量数量 `D` 相匹配。
  - `evaluate_objective` 函数必须能够接受一个D维向量并返回一个M维向量。
  - 种群矩阵 `f` 的前D列为决策变量，后M列为对应的目标函数值，便于后续的交叉、变异和选择操作。

---

### `Main.m` 文件

```matlab
% Main.m
% 非支配排序的蜣螂算法（NSDBO）的主程序
% 该程序初始化种群，运行NSDBO算法，并计算相关评价指标
% 输出:
%   Obtained_Pareto - 通过NSDBO算法获得的Pareto前沿解集
%   X               - Pareto前沿解集对应的决策变量位置

close all;    % 关闭所有打开的图形窗口
clear;        % 清除工作区中的所有变量
clc;          % 清空命令窗口

%% 设置测试问题
TestProblem = 31;                % 测试问题编号（范围1-47）
MultiObj = GetFunInfo(TestProblem); % 获取测试问题的详细信息
MultiObjFnc = MultiObj.name;     % 获取测试问题的名称

%% 设置算法参数
params.Np = 300;        % 种群规模（Population size）
params.Nr = 300;        % 仓库规模（Repository size）
params.maxgen = 300;    % 最大代数（Maximum number of generations）

numOfObj = MultiObj.numOfObj; % 目标函数个数
D = MultiObj.nVar;            % 决策变量维数

%% 运行NSDBO算法
f = NSDBO(params, MultiObj);     % 执行NSDBO算法，返回最终仓库中的个体

%% 提取结果
X = f(:, 1:D);                        % 提取仓库中个体的决策变量位置
Obtained_Pareto = f(:, D+1 : D+numOfObj); % 提取仓库中个体的目标函数值

%% 计算评价指标
if isfield(MultiObj, 'truePF')       % 判断是否存在真实Pareto前沿
    True_Pareto = MultiObj.truePF;    % 获取真实Pareto前沿
    
    %% 计算评价指标
    % ResultData的值分别是IGD、GD、HV、Spacing
    % 其中HV（超体积）越大越好，其他指标（IGD、GD、Spacing）越小越好
    ResultData = [
        IGD(Obtained_Pareto, True_Pareto),     % 反向广义差距
        GD(Obtained_Pareto, True_Pareto),      % 广义差距
        HV(Obtained_Pareto, True_Pareto),      % 超体积
        Spacing(Obtained_Pareto)               % 解集分布间距
    ];
else
    % 如果没有真实Pareto前沿，只计算Spacing
    % Spacing越小说明解集分布越均匀
    ResultData = Spacing(Obtained_Pareto);    % 计算Spacing
end

%% 显示结果
disp('Repository fitness values are stored in Obtained_Pareto');
disp('Repository particles positions are stored in X');
```

**注释说明：**
- **程序功能**：`Main.m` 是非支配排序的蜣螂算法（NSDBO）的主程序。该程序负责初始化种群，执行NSDBO算法，提取结果，并计算相关的评价指标（如IGD、GD、HV、Spacing）。
- **主要步骤**：
  1. **初始化环境**：
     - 关闭所有图形窗口，清除工作区变量，并清空命令窗口，确保程序在一个干净的环境中运行。
  2. **设置测试问题**：
     - 选择一个测试问题编号（范围1-47），通过 `GetFunInfo` 函数获取该测试问题的详细信息，包括目标函数数量、决策变量维数、变量范围、真实Pareto前沿等。
  3. **设置算法参数**：
     - 定义种群规模（`Np`）、仓库规模（`Nr`）和最大代数（`maxgen`）等参数。
     - 获取目标函数个数和决策变量维数。
  4. **运行NSDBO算法**：
     - 调用 `NSDBO` 函数，传入参数和测试问题信息，执行算法并获取最终仓库中的个体。
  5. **提取结果**：
     - 从算法输出中提取决策变量位置（`X`）和对应的目标函数值（`Obtained_Pareto`）。
  6. **计算评价指标**：
     - 如果存在真实Pareto前沿（`truePF`），则计算包括反向广义差距（IGD）、广义差距（GD）、超体积（HV）和解集分布间距（Spacing）在内的多个评价指标。
     - 如果不存在真实Pareto前沿，则仅计算解集分布间距（Spacing）。
     - 评价指标说明：
       - **IGD（反向广义差距）**：衡量真实Pareto前沿与种群解集的覆盖情况，值越小越好。
       - **GD（广义差距）**：衡量种群解集与真实Pareto前沿的收敛程度，值越小越好。
       - **HV（超体积）**：衡量种群解集在目标空间中所覆盖的体积，值越大越好。
       - **Spacing（解集分布间距）**：衡量种群解集的分布均匀性，值越小越好。
  7. **显示结果**：
     - 在命令窗口显示仓库中个体的目标函数值和决策变量位置的存储信息。

- **注意事项**：
  - **测试问题编号**：确保 `TestProblem` 的值在1到47之间，对应存在的测试函数。
  - **真实Pareto前沿**：若选择的测试问题具有真实Pareto前沿（`truePF`），算法将计算更全面的评价指标。
  - **评价指标函数**：确保 `IGD`、`GD`、`HV` 和 `Spacing` 函数已经正确定义并可调用。
  - **结果解释**：
    - `Obtained_Pareto` 存储了通过NSDBO算法获得的Pareto前沿解集的目标函数值。
    - `X` 存储了对应的决策变量位置，便于进一步分析或可视化。

---

### `non domination sort mod.m` 文件

```matlab
% non_domination_sort_mod.m
% 非支配排序修改版，用于对种群进行非支配排序并计算拥挤距离
% 参考自NSGA-II，版权所有。
% 输入:
%   x - 种群矩阵，每一行代表一个个体，包含决策变量和目标函数值
%   M - 目标函数的数量
%   D - 决策变量的数量
% 输出:
%   f - 排序后的种群矩阵，包含原始数据、排名和拥挤距离

function f = non_domination_sort_mod(x, M, D)

    %% 函数说明
    % 该函数基于非支配排序对当前种群进行排序。
    % 所有位于第一前沿的个体被赋予排名1，第二前沿的个体被赋予排名2，以此类推。
    % 排名分配后，计算每个前沿中个体的拥挤距离。

    %% 获取种群个体数
    [N, ~] = size(x);  % N为种群中的个体数量

    %% 初始化前沿编号
    front = 1;  % 初始前沿编号为1

    %% 初始化前沿结构体
    % F(front).f 存储第front前沿的个体索引
    F(front).f = [];
    individual = [];  % 初始化个体结构体数组，用于存储支配关系

    %% 非支配排序
    % 对种群中的每个个体进行非支配排序，确定其所属前沿

    for i = 1 : N
        % 初始化个体i的支配计数和被支配集合
        individual(i).n = 0;    % 被支配的个体数量
        individual(i).p = [];    % 支配个体的索引集合

        for j = 1 : N
            if i == j
                continue;  % 跳过自身比较
            end

            % 比较个体i和个体j的目标函数值
            dom_less = 0;    % i在某个目标上优于j
            dom_equal = 0;   % i和j在某个目标上相等
            dom_more = 0;    % i在某个目标上劣于j

            for k = 1 : M
                if (x(i, D + k) < x(j, D + k))
                    dom_less = dom_less + 1;
                elseif (x(i, D + k) == x(j, D + k))
                    dom_equal = dom_equal + 1;
                else
                    dom_more = dom_more + 1;
                end
            end

            if dom_less == 0 && dom_equal ~= M
                % 如果j在所有目标上不劣于i，且至少有一个目标优于i
                individual(i).n = individual(i).n + 1;
            elseif dom_more == 0 && dom_equal ~= M
                % 如果i在所有目标上不劣于j，且至少有一个目标优于j
                individual(i).p = [individual(i).p j];
            end
        end   % 结束j循环

        if individual(i).n == 0
            % 如果个体i不被任何个体支配，则属于第一前沿
            x(i, M + D + 1) = 1;      % 记录排名为1
            F(front).f = [F(front).f i];  % 添加到第一前沿
        end
    end % 结束i循环

    %% 寻找后续前沿
    while ~isempty(F(front).f)
        Q = [];  % 初始化下一个前沿的个体集合

        for i = 1 : length(F(front).f)
            p = F(front).f(i);  % 当前前沿中的个体索引

            if ~isempty(individual(p).p)
                for j = 1 : length(individual(p).p)
                    q = individual(p).p(j);  % 被p支配的个体索引
                    individual(q).n = individual(q).n - 1;  % 被支配计数减1

                    if individual(q).n == 0
                        % 如果q不再被任何个体支配，则属于下一个前沿
                        x(q, M + D + 1) = front + 1;  % 记录排名
                        Q = [Q q];  % 添加到下一个前沿集合
                    end
                end
            end
        end

        front = front + 1;      % 前沿编号加1
        F(front).f = Q;         % 更新当前前沿为下一个前沿
    end

    %% 根据前沿编号排序种群
    sorted_based_on_front = sortrows(x, M + D + 1);  % 按排名排序
    current_index = 0;  % 当前索引初始化

    %% 拥挤距离计算
    % 计算每个前沿中个体的拥挤距离，用于保持种群的多样性

    for front = 1 : (length(F) - 1)
        y = [];  % 当前前沿的个体数据
        previous_index = current_index + 1;

        % 提取当前前沿的个体
        for i = 1 : length(F(front).f)
            y(i, :) = sorted_based_on_front(current_index + i, :);
        end
        current_index = current_index + i;  % 更新当前索引

        % 对每个目标函数进行处理
        for i = 1 : M
            % 按第i个目标函数值排序当前前沿的个体
            [sorted_based_on_objective, index_of_objectives] = sortrows(y, D + i);

            % 获取当前目标的最大值和最小值
            f_max = sorted_based_on_objective(end, D + i);
            f_min = sorted_based_on_objective(1, D + i);

            % 为边界个体分配无限大的拥挤距离
            y(index_of_objectives(end), M + D + 1 + i) = Inf;
            y(index_of_objectives(1), M + D + 1 + i) = Inf;

            % 计算中间个体的拥挤距离
            for j = 2 : (length(index_of_objectives) - 1)
                next_obj = sorted_based_on_objective(j + 1, D + i);
                previous_obj = sorted_based_on_objective(j - 1, D + i);

                if (f_max - f_min == 0)
                    % 避免除以零
                    y(index_of_objectives(j), M + D + 1 + i) = Inf;
                else
                    % 计算拥挤距离
                    y(index_of_objectives(j), M + D + 1 + i) = ...
                        (next_obj - previous_obj) / (f_max - f_min);
                end
            end
        end

        % 累加每个个体在所有目标上的拥挤距离
        distance = zeros(length(F(front).f), 1);
        for i = 1 : M
            distance = distance + y(:, M + D + 1 + i);
        end
        y(:, M + D + 2) = distance;  % 添加拥挤距离列
        y = y(:, 1 : M + D + 2);    % 保留前M+D+2列
        z(previous_index : current_index, :) = y;  % 存储到排序后的种群
    end

    f = z();  % 返回排序和拥挤距离计算后的种群

end

%% 辅助函数

% Slice 函数
% 对给定点集进行切片操作
% 输入:
%   pl        - 点集
%   k         - 当前维度
%   RefPoint  - 参考点
% 输出:
%   S         - 切片结果

function S = Slice(pl, k, RefPoint)
    p  = Head(pl);    % 获取点集的第一个点
    pl = Tail(pl);    % 获取点集的剩余部分
    ql = [];          % 初始化切片后的点集
    S  = {};          % 初始化切片结果集合

    while ~isempty(pl)
        ql  = Insert(p, k + 1, ql);  % 插入点到切片集合
        p_  = Head(pl);               % 获取下一个点
        cell_(1,1) = {abs(p(k) - p_(k))};  % 计算当前维度的差值
        cell_(1,2) = {ql};                  % 关联切片后的点集
        S   = Add(cell_, S);                % 添加到切片结果集合
        p   = p_;                            % 更新当前点
        pl  = Tail(pl);                      % 更新点集
    end

    ql = Insert(p, k + 1, ql);  % 插入最后一个点
    cell_(1,1) = {abs(p(k) - RefPoint(k))};  % 计算与参考点的差值
    cell_(1,2) = {ql};                        % 关联切片后的点集
    S  = Add(cell_, S);                      % 添加到切片结果集合
end

% Insert 函数
% 将点插入到切片集合中，并处理点的顺序和支配关系
% 输入:
%   p  - 当前点
%   k  - 当前维度
%   pl - 切片后的点集
% 输出:
%   ql - 更新后的切片点集

function ql = Insert(p, k, pl)
    flag1 = 0;
    flag2 = 0;
    ql    = [];
    hp    = Head(pl);  % 获取切片点集的第一个点

    % 将小于当前点的点加入到切片点集中
    while ~isempty(pl) && hp(k) < p(k)
        ql = [ql; hp];
        pl = Tail(pl);
        hp = Head(pl);
    end

    ql = [ql; p];  % 插入当前点

    m  = length(p);  % 点的维数

    while ~isempty(pl)
        q = Head(pl);  % 获取下一个点

        for i = k : m
            if p(i) < q(i)
                flag1 = 1;
            else
                if p(i) > q(i)
                    flag2 = 1;
                end
            end
        end

        % 如果当前点p不完全支配点q，则保留点q
        if ~(flag1 == 1 && flag2 == 0)
            ql = [ql; Head(pl)];
        end

        pl = Tail(pl);  % 更新切片点集
    end
end

% Head 函数
% 获取点集的第一个点
% 输入:
%   pl - 点集
% 输出:
%   p  - 第一个点

function p = Head(pl)
    if isempty(pl)
        p = [];
    else
        p = pl(1, :);  % 返回第一个点
    end
end

% Tail 函数
% 获取点集的剩余部分（去除第一个点）
% 输入:
%   pl - 点集
% 输出:
%   ql - 剩余点集

function ql = Tail(pl)
    if size(pl, 1) < 2
        ql = [];
    else
        ql = pl(2:end, :);  % 返回除第一个点外的所有点
    end
end

% Add 函数
% 将切片结果添加到切片集合中，处理权重的累加
% 输入:
%   cell_ - 当前切片的权重和点集
%   S     - 现有的切片集合
% 输出:
%   S_    - 更新后的切片集合

function S_ = Add(cell_, S)
    n = size(S, 1);  % 当前切片集合的大小
    m = 0;
    for k = 1 : n
        if isequal(cell_(1,2), S(k,2))
            % 如果当前点集已存在于切片集合中，则累加权重
            S(k,1) = {cell2mat(S(k,1)) + cell2mat(cell_(1,1))};
            m = 1;
            break;
        end
    end
    if m == 0
        % 如果当前点集不存在于切片集合中，则添加新的切片
        S(n + 1, :) = cell_(1, :);
    end
    S_ = S;  % 返回更新后的切片集合
end
```

---

**注释说明：**

1. **函数功能概述：**
   - `non_domination_sort_mod` 函数用于对种群进行非支配排序，并计算每个个体的拥挤距离。该函数基于NSGA-II的非支配排序方法，能够将种群中的个体分配到不同的前沿（front）中，并在每个前沿内通过拥挤距离保持种群的多样性。

2. **主要步骤：**
   
   - **初始化阶段：**
     - 获取种群中个体的数量 `N`。
     - 初始化前沿编号 `front` 为1。
     - 初始化前沿结构体 `F(front).f` 用于存储属于当前前沿的个体索引。
     - 初始化个体结构体数组 `individual`，用于存储每个个体的支配计数和被支配集合。

   - **非支配排序阶段：**
     - 对种群中的每个个体 `i`：
       - 初始化其被支配计数 `individual(i).n` 为0，支配集合 `individual(i).p` 为空。
       - 与种群中每个其他个体 `j` 比较目标函数值：
         - 统计在每个目标维度上，个体 `i` 相对于 `j` 的优势（`dom_less`）、相等（`dom_equal`）和劣势（`dom_more`）。
         - 根据比较结果更新支配计数和被支配集合：
           - 如果 `i` 不被任何个体支配，则将其归入当前前沿，并赋予排名1。
           - 如果 `i` 支配 `j`，则将 `j` 添加到 `i` 的支配集合中。
     - 寻找后续前沿：
       - 循环处理每个前沿，逐步构建所有前沿。
       - 对于当前前沿中的每个个体，遍历其支配集合，更新被支配个体的支配计数。
       - 如果某个被支配个体的支配计数减为0，则将其归入下一个前沿。
   
   - **排序阶段：**
     - 根据前沿编号对种群进行排序，确保前沿优先。

   - **拥挤距离计算阶段：**
     - 对每个前沿（从前到后）：
       - 对每个目标函数，按照目标值对个体进行排序。
       - 为边界个体（最小值和最大值）赋予无限大的拥挤距离。
       - 对于中间个体，计算其在当前目标上的拥挤距离。
       - 累加所有目标上的拥挤距离，得到每个个体的总拥挤距离。
     - 最终返回包含排名和拥挤距离信息的种群矩阵 `f`。

3. **辅助函数说明：**
   
   - **`Slice` 函数：**
     - 对给定点集 `pl` 进行切片操作，基于当前维度 `k` 和参考点 `RefPoint`，将点集划分为不同的区域。
     - 返回切片结果 `S`，用于进一步的拥挤距离计算。
   
   - **`Insert` 函数：**
     - 将当前点 `p` 插入到切片集合 `ql` 中，确保切片集合中的点按目标维度有序，并处理支配关系。
     - 返回更新后的切片集合 `ql`。
   
   - **`Head` 函数：**
     - 获取点集 `pl` 的第一个点。如果点集为空，返回空数组。
   
   - **`Tail` 函数：**
     - 获取点集 `pl` 的剩余部分（去除第一个点）。如果点集只有一个点，返回空数组。
   
   - **`Add` 函数：**
     - 将当前切片结果 `cell_` 添加到切片集合 `S` 中。
     - 如果切片集合中已存在相同的点集，则累加其权重；否则，添加新的切片。
     - 返回更新后的切片集合 `S_`。

4. **具体代码注释：**
   
   - **非支配排序部分：**
     ```matlab
     for i = 1 : N
         % 初始化个体i的被支配计数和支配集合
         individual(i).n = 0;
         individual(i).p = [];
         for j = 1 : N
             dom_less = 0;
             dom_equal = 0;
             dom_more = 0;
             for k = 1 : M
                 if (x(i,D + k) < x(j,D + k))
                     dom_less = dom_less + 1;
                 elseif (x(i,D + k) == x(j,D + k))
                     dom_equal = dom_equal + 1;
                 else
                     dom_more = dom_more + 1;
                 end
             end
             if dom_less == 0 && dom_equal ~= M
                 % 个体j支配个体i
                 individual(i).n = individual(i).n + 1;
             elseif dom_more == 0 && dom_equal ~= M
                 % 个体i支配个体j
                 individual(i).p = [individual(i).p j];
             end
         end
         if individual(i).n == 0
             % 个体i属于第一前沿
             x(i,M + D + 1) = 1;
             F(front).f = [F(front).f i];
         end
     end
     ```

   - **寻找后续前沿部分：**
     ```matlab
     while ~isempty(F(front).f)
         Q = [];  % 初始化下一个前沿的个体集合
         for i = 1 : length(F(front).f)
             if ~isempty(individual(F(front).f(i)).p)
                 for j = 1 : length(individual(F(front).f(i)).p)
                     q = individual(F(front).f(i)).p(j);
                     individual(q).n = individual(q).n - 1;
                     if individual(q).n == 0
                         % 个体q属于下一个前沿
                         x(q,M + D + 1) = front + 1;
                         Q = [Q q];
                     end
                 end
             end
         end
         front =  front + 1;       % 前沿编号加1
         F(front).f = Q;            % 更新当前前沿为下一个前沿
     end
     ```

   - **拥挤距离计算部分：**
     ```matlab
     sorted_based_on_front = sortrows(x,M+D+1); % 按照前沿编号排序
     current_index = 0;

     for front = 1 : (length(F) - 1)
         y = [];
         previous_index = current_index + 1;
         for i = 1 : length(F(front).f)
             y(i,:) = sorted_based_on_front(current_index + i,:);
         end
         current_index = current_index + i;

         for i = 1 : M
             % 按照第i个目标函数排序
             [sorted_based_on_objective, index_of_objectives] = sortrows(y,D + i);       
             f_max = sorted_based_on_objective(length(index_of_objectives), D + i);
             f_min = sorted_based_on_objective(1, D + i);
             
             % 为边界个体赋予无限大的拥挤距离
             y(index_of_objectives(length(index_of_objectives)),M + D + 1 + i) = Inf;
             y(index_of_objectives(1),M + D + 1 + i) = Inf;
             
             for j = 2 : length(index_of_objectives) - 1
                 next_obj  = sorted_based_on_objective(j + 1,D + i);
                 previous_obj  = sorted_based_on_objective(j - 1,D + i);
                 if (f_max - f_min == 0)
                     y(index_of_objectives(j),M + D + 1 + i) = Inf;
                 else
                     y(index_of_objectives(j),M + D + 1 + i) = ...
                         (next_obj - previous_obj)/(f_max - f_min);
                 end
             end
         end

         % 累加每个个体在所有目标上的拥挤距离
         distance = zeros(length(F(front).f),1);
         for i = 1 : M
             distance = distance + y(:,M + D + 1 + i);
         end
         y(:,M + D + 2) = distance;  % 添加拥挤距离列
         y = y(:,1 : M + D + 2);    % 保留前M+D+2列
         z(previous_index:current_index,:) = y;  % 存储到排序后的种群
     end
     f = z();  % 返回排序和拥挤距离计算后的种群
     ```

5. **参考文献：**
   
   - [1] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan, "A Fast Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182-197, 2002.
   - [2] N. Srinivas and Kalyanmoy Deb, "Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms," *Evolutionary Computation*, vol. 2, no. 3, pp. 221-248, 1994.

---

**详细说明：**

1. **函数功能概述：**
   - `non_domination_sort_mod` 函数用于对种群进行非支配排序，将种群中的个体分配到不同的前沿（front）中。前沿1包含不被任何其他个体支配的个体，前沿2包含被前沿1中的个体支配但不被其他前沿2及以后的个体支配的个体，依此类推。
   - 排名分配完成后，函数还计算每个个体在其所属前沿中的拥挤距离（Crowding Distance），用于保持种群的多样性。

2. **主要步骤：**
   
   - **非支配排序阶段：**
     - **初始化支配计数和支配集合：**
       - 对于种群中的每个个体 `i`，初始化其被支配计数 `individual(i).n` 为0，支配集合 `individual(i).p` 为空。
     - **比较支配关系：**
       - 将种群中的每个个体 `i` 与其他所有个体 `j` 进行比较，基于其目标函数值确定支配关系：
         - 如果个体 `j` 在所有目标上不劣于 `i`，且至少在一个目标上优于 `i`，则增加 `i` 的被支配计数。
         - 如果个体 `i` 在所有目标上不劣于 `j`，且至少在一个目标上优于 `j`，则将 `j` 添加到 `i` 的支配集合中。
     - **分配前沿编号：**
       - 如果个体 `i` 的被支配计数为0，说明其不被任何其他个体支配，属于当前前沿，将其加入当前前沿 `F(front).f` 并赋予前沿编号1。
     - **迭代寻找后续前沿：**
       - 对于当前前沿中的每个个体，遍历其支配集合中的个体，减少这些被支配个体的被支配计数。
       - 如果某个被支配个体的被支配计数减为0，则说明其不再被任何前沿中的个体支配，将其加入下一个前沿。
       - 重复此过程，直到所有个体被分配到相应的前沿中。

   - **排序阶段：**
     - 根据前沿编号对种群进行排序，确保前沿优先。即前沿1的个体排在前面，前沿2的个体排在其后，以此类推。

   - **拥挤距离计算阶段：**
     - 对于每个前沿中的个体，计算其在各个目标上的拥挤距离，用于衡量个体在目标空间中的分布情况，保持种群的多样性。
     - **步骤如下：**
       - 对每个目标函数，按照目标值对前沿中的个体进行排序。
       - 为边界个体（最小值和最大值）赋予无限大的拥挤距离，以确保边界解被保留。
       - 对于中间个体，计算其在当前目标上的拥挤距离，公式为：
         \[
         \text{距离} = \frac{\text{下一个个体的目标值} - \text{前一个个体的目标值}}{\text{目标的最大值} - \text{目标的最小值}}
         \]
       - 累加所有目标上的拥挤距离，得到每个个体的总拥挤距离。

3. **辅助函数说明：**
   
   - **`Slice` 函数：**
     - 对给定点集 `pl` 进行切片操作，基于当前维度 `k` 和参考点 `RefPoint`，将点集划分为不同的区域。
     - 具体来说，函数通过插入点并计算差值，生成切片结果集合 `S`，用于后续的拥挤距离计算。

   - **`Insert` 函数：**
     - 将当前点 `p` 插入到切片集合 `ql` 中，确保切片集合中的点按目标维度有序，并处理支配关系。
     - 如果当前点 `p` 不完全支配 `pl` 中的某个点，则保留该点，确保切片结果的正确性。

   - **`Head` 和 `Tail` 函数：**
     - `Head(pl)` 函数返回点集 `pl` 的第一个点。
     - `Tail(pl)` 函数返回点集 `pl` 的剩余部分（去除第一个点）。

   - **`Add` 函数：**
     - 将当前切片结果 `cell_` 添加到切片集合 `S` 中。
     - 如果切片集合中已存在相同的点集，则累加其权重；否则，添加新的切片。

4. **代码优化与性能考虑：**
   
   - **效率提升：**
     - 通过预先计算和存储支配关系，避免重复比较，提高了非支配排序的效率。
   
   - **拥挤距离的计算：**
     - 采用排序和差值计算的方式，有效地计算了拥挤距离，确保种群的多样性。
   
   - **内存管理：**
     - 使用结构体数组 `individual` 和前沿结构体 `F`，有助于管理和存储个体的支配关系和前沿编号。

5. **返回结果说明：**
   
   - **`f`**：排序和拥挤距离计算后的种群矩阵，包含原始数据、排名和拥挤距离。具体来说，矩阵的每一行包含：
     - 前D列：决策变量值。
     - 接下来的M列：目标函数值。
     - 第M+D+1列：前沿编号（排名）。
     - 后续M列：每个目标上的拥挤距离。
     - 最后一列：总拥挤距离。

6. **注意事项：**
   
   - **输入矩阵 `x` 的格式**：
     - 确保输入矩阵 `x` 的每一行包含D个决策变量和M个目标函数值。
   
   - **索引和维度**：
     - 在代码中，目标函数值从第D+1列开始，到第D+M列结束。
     - 确保索引和维度的计算正确，避免数组越界或错误的值赋予。
   
   - **性能优化**：
     - 对于大规模种群和高维目标，非支配排序的计算可能会比较耗时。可以考虑进一步优化代码，例如使用向量化操作或并行计算。

---

**总结：**

`non_domination_sort_mod.m` 文件实现了基于非支配排序的种群排序和拥挤距离计算，主要用于多目标优化算法中，如NSGA-II和NSDBO。通过详细的注释，您可以更好地理解代码的逻辑和实现细节，并根据需要进行修改和优化。

---

### `NSDBO.m` 文件

```matlab
% NSDBO.m
% 非支配排序蜣螂优化算法（NSDBO）的主函数
% 该函数初始化种群，执行优化过程，并计算评价指标
% 输入:
%   params   - 算法参数结构体，包含种群规模、仓库规模、最大代数等
%   MultiObj - 多目标优化问题的信息结构体，包括目标函数、变量范围等
% 输出:
%   f - 最终仓库中的个体，包含决策变量和目标函数值

function f = NSDBO(params, MultiObj)
    tic  % 开始计时
    
    % 获取问题名称、目标函数个数、决策变量个数及变量范围
    name = MultiObj.name;
    numOfObj = MultiObj.numOfObj;  % 目标函数个数
    evaluate_objective = MultiObj.fun;  % 目标函数句柄
    D = MultiObj.nVar;  % 决策变量维数
    LB = MultiObj.var_min;  % 决策变量下界
    UB = MultiObj.var_max;  % 决策变量上界
    
    % 获取算法参数
    Max_iteration = params.maxgen;  % 最大代数
    SearchAgents_no = params.Np;  % 种群规模
    ishow = 1;  % 显示频率
    Nr = params.Nr;  % 仓库规模
    
    % 初始化种群（决策变量和目标函数值）
    chromosome = initialize_variables(SearchAgents_no, numOfObj, D, LB, UB, evaluate_objective);
    
    % 对初始种群进行非支配排序
    intermediate_chromosome = non_domination_sort_mod(chromosome, numOfObj, D);
    
    % 替换仓库中的个体
    Pop = replace_chromosome(intermediate_chromosome, numOfObj, D, Nr);
    
    M = numOfObj;  % 目标函数个数
    K = D + M;  % 决策变量和目标函数的总维数
    
    % 提取种群的决策变量和目标函数值
    POS = Pop(:, 1:K+1);  % 包含排名的种群矩阵
    POS_ad = POS(:, 1:K);  % 适应度存储，用于更新位置
    
    % 初始化新位置矩阵
    newPOS = zeros(SearchAgents_no, K);
    
    % 检查种群中的支配关系
    DOMINATED = checkDomination(POS(:, D+1:D+M));
    
    % 移除被支配的个体，更新仓库
    Pop = POS(~DOMINATED, :);
    ERP = Pop(:, 1:K+1);  % 仓库中的个体
    
    %% 优化循环
    Iteration = 1;  % 当前代数初始化
    
    % 根据种群规模计算不同阶段的个体数量
    pNum1 = floor(SearchAgents_no * 0.2);  % 20%的个体
    pNum2 = floor(SearchAgents_no * 0.4);  % 40%的个体
    pNum3 = floor(SearchAgents_no * 0.63); % 63%的个体
    
    while Iteration <= Max_iteration  % 迭代直到达到最大代数
        leng = size(ERP, 1);  % 当前仓库中的个体数
        r2 = rand;  % 随机数
        
        % 更新前20%的个体
        for i = 1 : pNum1
            if (r2 < 0.9)
                r1 = rand;  % 随机数
                a = rand;  % 随机数
                if (a > 0.1)
                    a = 1;
                else
                    a = -1;
                end
                worse = ERP(randperm(leng, 1), 1:D);  % 随机选择一个较差的个体
                % 更新位置，基于当前个体与较差个体的差异
                newPOS(i, 1:D) = POS(i, 1:D) + 0.3 * abs(POS(i, 1:D) - worse) + a * 0.1 * POS_ad(i, 1:D); % 方程 (1)
            else
                aaa = randperm(180, 1);  % 随机选择一个角度
                if (aaa == 0 || aaa == 90 || aaa == 180)
                    newPOS(i, 1:D) = POS(i, 1:D);  % 不更新
                end
                theta = aaa * pi / 180;  % 转换为弧度
                % 更新位置，基于角度的变化
                newPOS(i, 1:D) = POS(i, 1:D) + tan(theta) .* abs(POS(i, 1:D) - POS_ad(i, 1:D));    % 方程 (2)
            end
        end
        
        % 计算收敛因子R
        R = 1 - Iteration / Max_iteration;
        
        % 选择最佳个体用于更新位置
        bestXX = ERP(randperm(leng, 1), 1:D);
        
        % 更新位置，根据收敛因子R
        % 方程 (3)
        Xnew1 = bestXX .* (1 - R);
        Xnew2 = bestXX .* (1 + R);
        Xnew1 = bound(Xnew1, UB, LB);  % 越界判断
        Xnew2 = bound(Xnew2, UB, LB);  % 越界判断
        
        bestX = ERP(randperm(leng, 1), 1:D);
        % 方程 (5)
        Xnew11 = bestX .* (1 - R);
        Xnew22 = bestX .* (1 + R);
        Xnew11 = bound(Xnew11, UB, LB);  % 越界判断
        Xnew22 = bound(Xnew22, UB, LB);  % 越界判断
        
        % 更新中间40%的个体
        for i = (pNum1 + 1) : pNum2  % 方程 (4)
            newPOS(i, 1:D) = bestXX + (rand(1, D) .* (POS(i, 1:D) - Xnew1) + rand(1, D) .* (POS(i, 1:D) - Xnew2));
        end
        
        % 更新中间23%的个体
        for i = pNum2 + 1 : pNum3  % 方程 (6)
            newPOS(i, 1:D) = POS(i, 1:D) + (randn(1) .* (POS(i, 1:D) - Xnew11) + (rand(1, D) .* (POS(i, 1:D) - Xnew22)));
        end
        
        % 更新剩余个体
        for j = pNum3 + 1 : SearchAgents_no  % 方程 (7)
            newPOS(j, 1:D) = bestX + randn(1, D) .* ((abs(POS(j, 1:D) - bestXX)) + (abs(POS(j, 1:D) - bestX))) / 2;
        end
        
        %% 计算新位置的目标函数值
        for i = 1 : SearchAgents_no
            newPOS(i, 1:D) = bound(newPOS(i, 1:D), UB, LB);  % 越界判断
            newPOS(i, D + 1 : K) = evaluate_objective(newPOS(i, 1:D));  % 计算目标函数值
            
            % 判断是否更新适应度存储
            dom_less = 0;
            dom_equal = 0;
            dom_more = 0;
            for k = 1 : M
                if (newPOS(i, D + k) < POS(i, D + k))
                    dom_less = dom_less + 1;
                elseif (newPOS(i, D + k) == POS(i, D + k))
                    dom_equal = dom_equal + 1;
                else
                    dom_more = dom_more + 1;
                end
            end
            
            if dom_more == 0 && dom_equal ~= M
                % 如果新个体在所有目标上不劣于当前个体，更新位置
                POS_ad(i, 1:K) = POS(i, 1:K);
                POS(i, 1:K) = newPOS(i, 1:K);
            else
                % 否则，更新适应度存储
                POS_ad(i, 1:K) = newPOS(i, 1:K);
            end
        end
        
        %% 非支配排序和仓库更新
        pos_com = [POS(:, 1:K) ; POS_ad];  % 合并当前位置和适应度存储
        intermediate_pos = non_domination_sort_mod(pos_com, M, D);  % 对合并后的种群进行非支配排序
        POS = replace_chromosome(intermediate_pos, M, D, Nr);  % 替换仓库中的个体
        
        DOMINATED = checkDomination(POS(:, D + 1 : D + M));  % 检查支配关系
        Pop = POS(~DOMINATED, :);  % 移除被支配的个体
        ERP = Pop(:, 1:K + 1);  % 更新仓库中的个体
        
        %% 显示迭代信息
        if rem(Iteration, ishow) == 0
            disp(['NSDBO Iteration ' num2str(Iteration) ': Number of solutions in the archive = ' num2str(size(ERP, 1))]);
        end
        Iteration = Iteration + 1;  % 增加迭代次数
        
        %% 绘制Pareto前沿
        h_fig = figure(1);  % 创建或选择图形窗口
        
        if (numOfObj == 2)
            pl_data = ERP(:, D + 1 : D + M);  % 提取用于绘图的数据
            POS_fit = sortrows(pl_data, 2);  % 按第二个目标函数排序
            figure(h_fig); 
            try delete(h_par); end  % 删除之前的点
            h_par = plot(POS_fit(:, 1), POS_fit(:, 2), 'or'); hold on;  % 绘制Pareto前沿点
            if (isfield(MultiObj, 'truePF'))
                try delete(h_pf); end  % 删除之前的真实Pareto前沿
                h_pf = plot(MultiObj.truePF(:, 1), MultiObj.truePF(:, 2), '.k'); hold on;  % 绘制真实Pareto前沿
            end
            title(name);  % 设置标题
            xlabel('f1'); ylabel('f2');  % 设置坐标轴标签
            drawnow;  % 更新图形
        end
        
        if (numOfObj == 3)
            pl_data = ERP(:, D + 1 : D + M);  % 提取用于绘图的数据
            POS_fit = sortrows(pl_data, 3);  % 按第三个目标函数排序
            figure(h_fig); 
            try delete(h_par); end  % 删除之前的点
            h_par = plot3(POS_fit(:, 1), POS_fit(:, 2), POS_fit(:, 3), 'or'); hold on;  % 绘制Pareto前沿点
            if (isfield(MultiObj, 'truePF'))
                try delete(h_pf); end  % 删除之前的真实Pareto前沿
                h_pf = plot3(MultiObj.truePF(:, 1), MultiObj.truePF(:, 2), MultiObj.truePF(:, 3), '.k'); hold on;  % 绘制真实Pareto前沿
            end
            title(name);  % 设置标题
            grid on;  % 显示网格
            xlabel('f1'); ylabel('f2'); zlabel('f3');  % 设置坐标轴标签
            drawnow;  % 更新图形
        end
    end
    toc  % 结束计时
    
    % 返回最终仓库中的个体
    f = ERP;
    hold off;
    
    % 添加图例
    if (isfield(MultiObj, 'truePF'))
        legend('NSDBO', 'TruePF');
    else
        legend('NSDBO');
    end
end

%% 辅助函数

% bound 函数
% 检查并修正个体的决策变量是否超出边界
% 输入:
%   a  - 决策变量向量
%   ub - 上界向量
%   lb - 下界向量
% 输出:
%   a  - 修正后的决策变量向量
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);  % 将超过上界的值设为上界
    a(a < lb) = lb(a < lb);  % 将低于下界的值设为下界
    % 以下两行代码用于在越界时重新随机生成值
    % a(a > ub) = rand(1).*(ub(a > ub) - lb(a > ub)) + lb(a > ub);
    % a(a < lb) = rand(1).*(ub(a < lb) - lb(a < lb)) + lb(a < lb);
end

% dominates 函数
% 判断个体x是否支配个体y
% 输入:
%   x - 个体x的目标函数值矩阵
%   y - 个体y的目标函数值矩阵
% 输出:
%   d - 支配结果向量，1表示x支配y，0表示不支配
function d = dominates(x, y)
    d = all(x <= y, 2) & any(x < y, 2);  % x在所有目标上不劣于y，且至少在一个目标上优于y
end

% checkDomination 函数
% 检查种群中个体的支配关系
% 输入:
%   fitness - 种群中个体的目标函数值矩阵
% 输出:
%   dom_vector - 支配向量，1表示被支配，0表示不被支配
function dom_vector = checkDomination(fitness)
    Np = size(fitness, 1);  % 种群规模
    dom_vector = zeros(Np, 1);  % 初始化支配向量
    
    % 生成所有可能的个体对排列
    all_perm = nchoosek(1:Np, 2);  % 生成所有2个个体的组合
    all_perm = [all_perm; [all_perm(:, 2) all_perm(:, 1)]];  % 添加反向排列
    
    % 判断哪些个体对满足x支配y
    d = dominates(fitness(all_perm(:, 1), :), fitness(all_perm(:, 2), :));
    
    % 找出被支配的个体索引
    dominated_particles = unique(all_perm(d == 1, 2));
    dom_vector(dominated_particles) = 1;  % 标记被支配的个体
end
```

---

### **详细说明**

#### 1. **函数功能概述**

- **`NSDBO` 函数**：
  - 实现了非支配排序蜣螂优化算法（NSDBO）。
  - 负责初始化种群，执行迭代优化过程，更新种群位置，并计算评价指标如IGD、GD、HV和Spacing。
  - 最终输出经过优化后的仓库中个体的决策变量和目标函数值。

- **辅助函数**：
  - **`bound`**：用于检查和修正个体的决策变量是否超出预定义的上下界。
  - **`dominates`**：判断一个个体是否支配另一个个体。
  - **`checkDomination`**：检查种群中所有个体的支配关系，标记被支配的个体。

#### 2. **主要步骤**

1. **初始化阶段**：
   - **获取问题信息**：从 `MultiObj` 结构体中提取问题名称、目标函数个数、决策变量个数和变量范围。
   - **设置算法参数**：包括种群规模 (`Np`)、仓库规模 (`Nr`)、最大代数 (`maxgen`) 和显示频率 (`ishow`)。
   - **初始化种群**：调用 `initialize_variables` 函数生成初始种群，包含决策变量和目标函数值。

2. **非支配排序和仓库更新**：
   - **非支配排序**：使用 `non_domination_sort_mod` 函数对种群进行非支配排序，分配前沿编号。
   - **替换仓库中的个体**：调用 `replace_chromosome` 函数将排序后的种群替换到仓库中，确保仓库中存储的是最优的非支配解集。
   - **检查支配关系**：通过 `checkDomination` 函数标记被支配的个体，并移除这些个体，仅保留非支配的解集。

3. **优化循环**：
   - **迭代控制**：通过 `while Iteration <= Max_iteration` 控制优化的代数。
   - **位置更新**：
     - 根据当前迭代次数和随机数，更新不同比例的个体位置。前20%、40%、63%和剩余个体使用不同的更新策略，基于现有解和随机扰动进行位置调整。
     - **位置更新方程**：
       - **方程 (1)**：基于当前个体与随机选择的较差个体之间的差异进行更新。
       - **方程 (2)**：基于随机角度的扰动进行更新。
       - **方程 (3)** 和 **方程 (5)**：基于最佳个体和收敛因子 `R` 的位置更新。
       - **方程 (4)**、**方程 (6)** 和 **方程 (7)**：基于随机权重和最佳个体进行混合更新。
   - **越界处理**：使用 `bound` 函数确保更新后的个体决策变量在合法范围内。
   - **目标函数值计算**：对更新后的个体计算目标函数值，并根据支配关系决定是否更新适应度存储。
   - **非支配排序和仓库更新**：再次对合并后的种群进行非支配排序，并更新仓库中的个体，确保仓库中存储的是当前最优的非支配解集。
   - **迭代信息显示**：根据 `ishow` 参数控制显示当前迭代的仓库解集数量。
   - **Pareto前沿绘制**：
     - 如果目标函数个数为2或3，则绘制当前仓库中的Pareto前沿。
     - 对于二维目标，使用 `plot` 函数绘制；对于三维目标，使用 `plot3` 函数绘制。
     - 如果存在真实Pareto前沿（`truePF`），则一并绘制真实Pareto前沿以进行对比。

4. **结束阶段**：
   - **计时结束**：使用 `toc` 记录算法运行时间。
   - **返回结果**：将最终仓库中的个体（包含决策变量和目标函数值）返回。
   - **图例添加**：根据是否存在真实Pareto前沿，添加相应的图例。

#### 3. **辅助函数说明**

- **`bound` 函数**：
  - **功能**：检查并修正个体的决策变量是否超出预定义的上下界。如果超出，则将其设为边界值。
  - **输入**：
    - `a`：决策变量向量。
    - `ub`：决策变量上界向量。
    - `lb`：决策变量下界向量。
  - **输出**：
    - `a`：修正后的决策变量向量。

- **`dominates` 函数**：
  - **功能**：判断一个个体是否支配另一个个体。
  - **输入**：
    - `x`：个体x的目标函数值矩阵。
    - `y`：个体y的目标函数值矩阵。
  - **输出**：
    - `d`：支配结果向量，1表示x支配y，0表示不支配。

- **`checkDomination` 函数**：
  - **功能**：检查种群中所有个体的支配关系，标记被支配的个体。
  - **输入**：
    - `fitness`：种群中个体的目标函数值矩阵。
  - **输出**：
    - `dom_vector`：支配向量，1表示被支配，0表示不被支配。

#### 4. **关键算法步骤解析**

- **非支配排序**：
  - 对种群中的每个个体，计算其被支配的数量和支配的个体集合。
  - 将不被任何个体支配的个体归入第一前沿（front 1）。
  - 依次处理每个前沿中的个体，更新其支配集合中的个体的被支配计数，将被支配计数为0的个体归入下一个前沿。
  
- **拥挤距离计算**：
  - 对每个前沿中的个体，计算其在每个目标函数上的拥挤距离。
  - 拥挤距离用于衡量个体在目标空间中的分布密度，保持种群的多样性。
  - 边界个体的拥挤距离设为无穷大，确保边界解的保留。

- **位置更新策略**：
  - **前20%的个体**：主要基于与较差个体的差异进行更新，增强搜索的局部性。
  - **中间40%的个体**：基于最佳个体和收敛因子进行混合更新，平衡全局和局部搜索。
  - **中间23%的个体**：引入随机扰动，增加搜索的多样性。
  - **剩余个体**：基于最佳个体的全局搜索策略，探索更广泛的目标空间。

#### 5. **注意事项**

- **算法参数设置**：
  - `Np`（种群规模）、`Nr`（仓库规模）和`maxgen`（最大代数）对算法性能有重要影响。需根据具体问题进行调参。
  
- **边界处理**：
  - `bound` 函数确保个体在更新后不超出决策变量的合法范围，是算法稳定性的关键。

- **非支配排序和拥挤距离**：
  - 非支配排序和拥挤距离计算是NSGA-II和NSDBO算法的核心部分，确保了种群的多样性和解的质量。

- **计算效率**：
  - 对于大规模种群和高维目标，非支配排序和拥挤距离计算可能会成为性能瓶颈。可考虑进一步优化算法或使用并行计算。

- **绘图和可视化**：
  - 算法在每代末期绘制当前的Pareto前沿，有助于实时观察算法的收敛情况和解集分布。

#### 6. **参考文献**

- [1] *Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan*, "A Fast Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182-197, 2002.
  
- [2] *N. Srinivas and Kalyanmoy Deb*, "Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms," *Evolutionary Computation*, vol. 2, no. 3, pp. 221-248, 1994.

---

### **补充说明**

1. **函数间调用关系**：
   - **`initialize_variables`**：用于初始化种群的决策变量和目标函数值。
   - **`non_domination_sort_mod`**：用于对种群进行非支配排序，并计算拥挤距离。
   - **`replace_chromosome`**：用于替换仓库中的个体，确保仓库中存储的是最优的非支配解集。
   - **`checkDomination`**：用于检查种群中个体的支配关系，标记被支配的个体。
   
2. **数据结构**：
   - **种群矩阵 `POS`**：包含个体的决策变量和目标函数值，以及前沿编号和拥挤距离。
   - **适应度存储 `POS_ad`**：用于记录个体的适应度，辅助位置更新。
   - **新位置矩阵 `newPOS`**：用于存储个体在当前代的更新位置。
   - **仓库矩阵 `ERP`**：存储非支配解集，作为最终输出结果。

3. **位置更新机制**：
   - **局部搜索**：通过与较差个体的差异更新，增强局部搜索能力。
   - **全局搜索**：基于最佳个体和随机扰动，探索更广泛的目标空间。
   - **混合策略**：结合不同的更新策略，平衡探索与开发，提高算法的搜索效率和解的质量。

4. **评价指标**：
   - **IGD（反向广义差距）**：衡量真实Pareto前沿与种群解集的覆盖情况。
   - **GD（广义差距）**：衡量种群解集与真实Pareto前沿的收敛程度。
   - **HV（超体积）**：衡量种群解集在目标空间中所覆盖的体积。
   - **Spacing（解集分布间距）**：衡量种群解集的分布均匀性。

---

### **总结**

`NSDBO.m` 文件实现了非支配排序蜣螂优化算法（NSDBO），通过初始化种群、执行迭代优化、进行非支配排序和拥挤距离计算，最终获得优化后的Pareto前沿解集。详细的中文注释帮助您理解每一部分代码的具体功能和作用，便于进一步的调试和优化。

---

### `replace_chromosome.m` 文件

```matlab
% replace_chromosome.m
% 替换染色体函数，用于根据排名和拥挤距离替换种群中的个体
% 参考自NSGA-II，版权所有。
% 输入:
%   intermediate_chromosome - 经过非支配排序和拥挤距离计算后的种群矩阵
%                            每一行包含决策变量、目标函数值、排名和拥挤距离
%   M                      - 目标函数的数量
%   D                      - 决策变量的数量
%   NP                     - 种群规模（Population Size）
% 输出:
%   f                      - 替换后的种群矩阵，大小为NP x (D + M + 2)
%                            每一行包含决策变量、目标函数值、排名和拥挤距离

function f = replace_chromosome(intermediate_chromosome, M, D, NP)

    %% 函数说明
    % 该函数根据个体的排名（Rank）和拥挤距离（Crowding Distance）替换种群中的个体。
    % 首先按排名对个体进行排序，依次添加每个前沿（Front）中的个体，
    % 直到达到种群规模NP。如果添加完整个前沿会超过NP，
    % 则根据拥挤距离从当前前沿中选择合适的个体填充到种群中。
    
    %% 按排名排序种群
    sorted_chromosome = sortrows(intermediate_chromosome, M + D + 1);
    % sorted_chromosome按第(M+D+1)列（排名）从小到大排序
    
    %% 获取最大排名
    max_rank = max(intermediate_chromosome(:, M + D + 1));
    % max_rank为当前种群中的最大排名值
    
    %% 初始化替换后的种群矩阵
    f = [];  % 初始化为空矩阵
    
    %% 逐前沿添加个体到种群
    previous_index = 0;  % 上一个前沿结束的索引
    for i = 1 : max_rank
        % 找到当前排名i的个体的最后一个索引
        current_index = find(sorted_chromosome(:, M + D + 1) == i, 1, 'last');
        
        if isempty(current_index)
            % 如果当前前沿没有个体，跳过
            continue;
        end
        
        % 判断添加当前前沿的个体是否会超过种群规模
        if (current_index > NP)
            % 计算剩余可添加的个体数量
            remaining = NP - previous_index;
            
            if remaining <= 0
                % 如果没有剩余空间，结束替换过程
                break;
            end
            
            % 提取当前前沿的所有个体
            temp_pop = sorted_chromosome(previous_index + 1 : current_index, :);
            
            % 按拥挤距离从高到低排序
            [~, temp_sort_index] = sort(temp_pop(:, M + D + 2), 'descend');
            % temp_sort_index是拥挤距离降序排列的索引
            
            % 从拥挤距离最高的个体开始添加，直到填满种群
            for j = 1 : remaining
                f = [f; temp_pop(temp_sort_index(j), :)];
                if j == remaining
                    break;
                end
            end
            
            % 填满种群后，结束替换过程
            return;
        elseif (current_index <= NP)
            % 如果添加当前前沿的所有个体不会超过种群规模，
            % 则将这些个体全部添加到替换后的种群中
            f = [f; sorted_chromosome(previous_index + 1 : current_index, :)];
        else
            % 如果添加当前前沿的所有个体会超过种群规模，
            % 则根据拥挤距离选择部分个体添加
            remaining = NP - previous_index;
            if remaining > 0
                temp_pop = sorted_chromosome(previous_index + 1 : current_index, :);
                [~, temp_sort_index] = sort(temp_pop(:, M + D + 2), 'descend');
                for j = 1 : remaining
                    f = [f; temp_pop(temp_sort_index(j), :)];
                    if j == remaining
                        break;
                    end
                end
            end
            return;
        end
        
        % 更新previous_index为当前前沿的最后一个索引
        previous_index = current_index;
    end
    
end
```

---

### **详细说明**

#### 1. **函数功能概述**

- **`replace_chromosome` 函数**：
  - 该函数用于根据个体的非支配排序排名（Rank）和拥挤距离（Crowding Distance）来替换种群中的个体，以保持种群的多样性和优良性。
  - 具体步骤包括：
    1. 按排名对种群进行排序。
    2. 依次添加每个前沿（Front）中的个体到替换后的种群中，直到达到种群规模NP。
    3. 如果添加完整个前沿会导致种群规模超过NP，则根据拥挤距离从当前前沿中选择个体填充到种群中。

#### 2. **主要步骤解析**

1. **按排名排序种群**：
   ```matlab
   sorted_chromosome = sortrows(intermediate_chromosome, M + D + 1);
   ```
   - 将`intermediate_chromosome`按第`(M+D+1)`列（排名）从小到大排序，确保排名较低（即更优）的个体排在前面。

2. **获取最大排名**：
   ```matlab
   max_rank = max(intermediate_chromosome(:, M + D + 1));
   ```
   - 计算当前种群中最高的排名值，用于确定需要遍历的前沿数量。

3. **初始化替换后的种群矩阵**：
   ```matlab
   f = [];  % 初始化为空矩阵
   ```
   - 用于存储最终替换后的种群。

4. **逐前沿添加个体到种群**：
   ```matlab
   previous_index = 0;  % 上一个前沿结束的索引
   for i = 1 : max_rank
       % 找到当前排名i的个体的最后一个索引
       current_index = find(sorted_chromosome(:, M + D + 1) == i, 1, 'last');
       
       if isempty(current_index)
           % 如果当前前沿没有个体，跳过
           continue;
       end
       
       % 判断添加当前前沿的个体是否会超过种群规模
       if (current_index > NP)
           % 计算剩余可添加的个体数量
           remaining = NP - previous_index;
           
           if remaining <= 0
               % 如果没有剩余空间，结束替换过程
               break;
           end
           
           % 提取当前前沿的所有个体
           temp_pop = sorted_chromosome(previous_index + 1 : current_index, :);
           
           % 按拥挤距离从高到低排序
           [~, temp_sort_index] = sort(temp_pop(:, M + D + 2), 'descend');
           
           % 从拥挤距离最高的个体开始添加，直到填满种群
           for j = 1 : remaining
               f = [f; temp_pop(temp_sort_index(j), :)];
               if j == remaining
                   break;
               end
           end
           
           % 填满种群后，结束替换过程
           return;
       elseif (current_index <= NP)
           % 如果添加当前前沿的所有个体不会超过种群规模，
           % 则将这些个体全部添加到替换后的种群中
           f = [f; sorted_chromosome(previous_index + 1 : current_index, :)];
       else
           % 如果添加当前前沿的所有个体会超过种群规模，
           % 则根据拥挤距离选择部分个体添加
           remaining = NP - previous_index;
           if remaining > 0
               temp_pop = sorted_chromosome(previous_index + 1 : current_index, :);
               [~, temp_sort_index] = sort(temp_pop(:, M + D + 2), 'descend');
               for j = 1 : remaining
                   f = [f; temp_pop(temp_sort_index(j), :)];
                   if j == remaining
                       break;
                   end
               end
           end
           return;
       end
       
       % 更新previous_index为当前前沿的最后一个索引
       previous_index = current_index;
   end
   ```
   
   - **逻辑说明**：
     - **遍历每个前沿**：
       - 对于每个前沿编号`i`，找到该前沿中所有个体的最后一个索引`current_index`。
     - **判断是否超过种群规模**：
       - 如果将当前前沿中的所有个体添加到替换后的种群中不会超过种群规模`NP`，则将这些个体全部添加。
       - 如果添加当前前沿中的所有个体会导致种群规模超过`NP`，则根据拥挤距离从当前前沿中选择个体，按拥挤距离从高到低排序，选择前`remaining`个个体填充到替换后的种群中，直到达到种群规模`NP`。
     - **终止条件**：
       - 一旦种群规模达到`NP`，则结束替换过程，返回替换后的种群`f`。

#### 3. **排序和拥挤距离的作用**

- **排名（Rank）**：
  - 用于确定个体的优劣，排名越低表示个体越优。
  - 先按排名筛选优质个体，再根据拥挤距离保持种群的多样性。

- **拥挤距离（Crowding Distance）**：
  - 用于在同一前沿内衡量个体的分布密度。
  - 拥挤距离越大，表示个体在目标空间中的位置越稀疏，能够保持解集的多样性。

#### 4. **注意事项**

- **输入矩阵格式**：
  - `intermediate_chromosome` 每一行应包含决策变量、目标函数值、排名（Rank）和拥挤距离（Crowding Distance）。
  - 具体来说，每一行的结构如下：
    ```
    [决策变量1, 决策变量2, ..., 决策变量D, 目标函数1, 目标函数2, ..., 目标函数M, Rank, Crowding Distance1, Crowding Distance2, ..., Crowding DistanceM]
    ```
  
- **拥挤距离列的位置**：
  - 在代码中，拥挤距离位于第`(M+D+2)`列。
  
- **种群规模限制**：
  - 确保`intermediate_chromosome`中的个体数不小于`NP`，否则在选择拥挤距离时可能会出现索引错误。

- **性能优化**：
  - 在处理大规模种群时，循环和动态数组拼接（如`f = [f; ...]`）可能会影响性能。可以考虑预分配矩阵大小或使用其他高效的数据结构。

#### 5. **参考文献**

- [1] *Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan*, "A Fast Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182-197, 2002.
  
- [2] *N. Srinivas and Kalyanmoy Deb*, "Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms," *Evolutionary Computation*, vol. 2, no. 3, pp. 221-248, 1994.

---

### **总结**

`replace_chromosome.m` 文件实现了根据非支配排序的排名和拥挤距离来替换种群中的个体，确保种群既具有优秀的解集，又保持了多样性。通过详细的中文注释，您可以更好地理解函数的逻辑和实现细节，便于进一步的调试和优化。

---

### `Spacing.m` 文件

```matlab
% Spacing.m
% 计算种群的间距（Spacing）指标
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
% 输出:
%   Score  - 种群的间距值，反映解集的分布均匀性

function Score = Spacing(PopObj)

    % 计算种群中每对个体之间的曼哈顿距离（Cityblock Distance）
    Distance = pdist2(PopObj, PopObj, 'cityblock');
    
    % 将距离矩阵的对角线元素（即个体与自身的距离）设为无穷大，避免在后续计算中被选中
    Distance(logical(eye(size(Distance, 1)))) = inf;
    
    % 对于每个个体，找到其与其他所有个体的最小距离
    minDistances = min(Distance, [], 2);
    
    % 计算所有个体的最小距离的标准差，作为间距指标
    % 标准差越小，说明解集分布越均匀
    Score = std(minDistances);
end
```

---

**注释说明：**

1. **函数功能概述：**
   - **`Spacing` 函数**用于计算种群的间距（Spacing）指标。该指标用于衡量种群解集的分布均匀性，标准差越小表示解集的分布越均匀，越有助于覆盖目标空间。

2. **参数说明：**
   - **输入参数：**
     - `PopObj`：种群中所有个体的目标函数值矩阵。矩阵的每一行代表一个个体在所有目标上的值。
       - 例如，对于一个二维目标问题，`PopObj` 可能是一个 `N x 2` 的矩阵，其中 `N` 是种群规模。
   - **输出参数：**
     - `Score`：种群的间距值。该值为所有个体最小距离的标准差，反映了解集的分布均匀性。
       - 标准差越小，表示个体之间的最小距离更加接近，解集分布越均匀。
       - 反之，标准差较大表示解集存在聚集或稀疏的现象，分布不够均匀。

3. **具体代码解析：**

   ```matlab
   % 计算种群中每对个体之间的曼哈顿距离（Cityblock Distance）
   Distance = pdist2(PopObj, PopObj, 'cityblock');
   ```
   - **功能**：使用 `pdist2` 函数计算种群中每对个体之间的曼哈顿距离（Cityblock Distance）。
     - **曼哈顿距离**定义为两个点在各个维度上的绝对距离之和，适用于衡量多维空间中个体之间的距离。
     - **返回值**：`Distance` 是一个 `N x N` 的矩阵，其中 `Distance(i,j)` 表示第 `i` 个个体与第 `j` 个个体之间的曼哈顿距离。

   ```matlab
   % 将距离矩阵的对角线元素（即个体与自身的距离）设为无穷大，避免在后续计算中被选中
   Distance(logical(eye(size(Distance, 1)))) = inf;
   ```
   - **功能**：将距离矩阵 `Distance` 的对角线元素（即每个个体与自身的距离）设为无穷大 `inf`。
     - **原因**：在后续计算每个个体的最小距离时，不希望个体自身的距离影响结果，因此将其设为 `inf` 确保不被选中。

   ```matlab
   % 对于每个个体，找到其与其他所有个体的最小距离
   minDistances = min(Distance, [], 2);
   ```
   - **功能**：对于种群中的每个个体，找到其与其他所有个体的最小距离。
     - **返回值**：`minDistances` 是一个 `N x 1` 的向量，其中 `minDistances(i)` 表示第 `i` 个个体与其他个体之间的最小曼哈顿距离。

   ```matlab
   % 计算所有个体的最小距离的标准差，作为间距指标
   % 标准差越小，说明解集分布越均匀
   Score = std(minDistances);
   ```
   - **功能**：计算所有个体的最小距离的标准差，作为种群的间距指标。
     - **解释**：
       - **标准差**：衡量数据的离散程度，标准差越小表示数据越集中，越大表示数据越分散。
       - 在此，`std(minDistances)` 表示所有个体最小距离的标准差。
         - **标准差小**：表示个体之间的最小距离差异较小，解集分布较为均匀。
         - **标准差大**：表示个体之间的最小距离差异较大，解集存在聚集或稀疏现象。

4. **应用场景：**
   - **多目标优化评价**：
     - 在多目标优化中，`Spacing` 指标常用于评估算法生成的解集在目标空间中的分布均匀性。
     - 通常与其他指标（如 IGD、HV）一起使用，以全面评估优化算法的性能。
   - **算法选择与调优**：
     - 通过计算不同算法或不同参数设置下的 `Spacing` 值，可以选择出能够生成更均匀解集的优化算法或最佳参数组合。

5. **注意事项：**
   - **距离计算方法**：
     - 本函数使用曼哈顿距离（Cityblock Distance）作为距离度量方法。如果需要使用其他距离度量（如欧几里得距离），可以修改 `pdist2` 函数的第三个参数。
   - **种群规模**：
     - 随着种群规模的增加，距离矩阵的计算量会显著增加，可能影响计算效率。对于大规模种群，考虑优化距离计算或使用近似方法。
   - **目标维数**：
     - 当目标函数维数较高时，距离计算和分布均匀性的评估可能会变得复杂。可以结合其他多样性指标共同评估种群的质量。

---

**总结：**

`Spacing.m` 文件实现了计算种群间距（Spacing）指标的功能，主要用于评估多目标优化算法生成的解集在目标空间中的分布均匀性。通过计算每个个体与其他个体的最小距离，并取这些最小距离的标准差作为间距指标，能够有效衡量解集的多样性。详细的中文注释帮助您理解函数的逻辑和实现细节，便于在多目标优化算法中应用和扩展。

---

### `testmop.m` 文件

```matlab
% testmop.m
% 测试多目标优化问题生成器
% 该脚本定义了多个常用的多目标优化测试问题，包括ZDT、DTLZ、WFG系列以及CEC2009的UF和CF函数。
% 通过输入测试问题的名称和维度，可以生成相应的测试问题结构体，包含目标函数、决策变量范围等信息。
%
% 输入:
%   testname    : (字符数组) 测试问题的名称，如 'zdt1', 'dtlz2', 'wfg1', 'uf1', 'cf1' 等。
%   dimension   : (整数) 决策变量的维数。
%                - 对于DTLZ问题，维数 = M - 1 + k；
%                - 对于WFG问题，维数 = l + k；
%
% 全局输入: 注意在选择对应的测试问题时，必须赋值以下全局变量，这些变量在函数中被标记为关键字 'global'。
%   动态问题:
%     itrCounter - 当前迭代次数
%     step       - 动态步长
%     window     - 动态窗口数量
%   WFG问题:
%     k - 与位置相关的参数数量
%     M - 目标函数数量
%     l - 与距离相关的参数数量
%   DTLZ问题:
%     M - 目标函数数量
%     k - 控制维数的参数
%
% 输出:
%   mop         : (结构体) 包含测试问题的详细信息
%                 - name   : 测试问题名称
%                 - od     : 目标维数（Objective Dimension）
%                 - pd     : 决策变量维数（Decision Dimension）
%                 - domain : 决策变量的边界约束（决策边界）
%                 - func   : 目标函数的句柄

function mop = testmop(testname, dimension)
    % 初始化测试问题结构体
    mop = struct('name',[],'od',[],'pd',[],'domain',[],'func',[]);
    
    % 动态调用对应的测试问题生成器函数
    eval(['mop=',testname,'(mop,',num2str(dimension),');']);
end

%% ----------Stationary Multi-Objective Benchmark----------
% ----------ZDT系列。参考文献：[2]----------- 

%% ZDT1 函数生成器
function p = zdt1(p, dim)
    p.name = 'ZDT1';
    p.pd = dim;        % 决策变量维数
    p.od = 2;          % 目标函数维数
    p.domain = [zeros(dim,1) ones(dim,1)]; % 决策变量范围 [0,1]^dim
    p.func = @evaluate; % 目标函数句柄
    
    % ZDT1 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        y(1) = x(1);
        su = sum(x) - x(1);
        g = 1 + 9 * su / (dim - 1);
        y(2) = g * (1 - sqrt(x(1) / g));
    end
end

%% ZDT2 函数生成器
function p = zdt2(p, dim)
    p.name = 'ZDT2';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % ZDT2 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 9 * sum(x(2:dim)) / (dim - 1);
        y(1) = x(1);
        y(2) = g * (1 - (x(1)/g)^2);
    end
end

%% ZDT3 函数生成器
function p = zdt3(p, dim)
    p.name = 'ZDT3';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % ZDT3 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 9 * sum(x(2:dim)) / (dim - 1);
        y(1) = x(1);
        y(2) = g * (1 - sqrt(x(1)/g) - (x(1)/g) * sin(10 * pi * x(1)));
    end
end

%% ZDT4 函数生成器
function p = zdt4(p, dim)
    p.name = 'ZDT4';
    p.pd = dim;
    p.od = 2;
    % 决策变量范围：
    % x1 ∈ [0,1]
    % x2, ..., xdim ∈ [-5,5]
    p.domain = [0 * ones(dim,1) 1 * ones(dim,1)];
    p.domain(1,1) = 0; % x1的下界
    p.domain(1,2) = 1; % x1的上界
    p.func = @evaluate;
    
    % ZDT4 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 10 * (10 - 1);
        for i = 2:10
            g = g + x(i)^2 - 10 * cos(4 * pi * x(i));
        end
        y(1) = x(1);
        y(2) = g * (1 - sqrt(x(1)/g));
    end
end

%% ZDT6 函数生成器
function p = zdt6(p, dim)
    p.name = 'ZDT6';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % ZDT6 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 9 * (sum(x(2:dim)) / (dim - 1))^0.25;
        y(1) = 1 - exp(-4 * x(1)) * sin(6 * pi * x(1))^6;
        y(2) = g * (1 - (y(1)/g)^2);
    end
end

%% --------------DTLZ 基准测试-------参考文献：[3]----
% DTLZ系列问题定义

%% DTLZ1 函数生成器
% 建议参数：k = 5 
function p = DTLZ1(p, dim)
    global M k;
    p.name = 'DTLZ1';
    p.pd = dim;      % 决策变量维数
    p.od = M;        % 目标函数维数
    p.domain = [zeros(dim,1) ones(dim,1)]; % 决策变量范围 [0,1]^dim
    p.func = @evaluate;
    
    % DTLZ1 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 5 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = 100 * (k + sum((xm - 0.5).^2 - cos(20 * pi * (xm - 0.5)), 1));

        % 计算目标函数
        f(1,:) = 0.5 * prod(x(1:M-1,:), 1) .* (1 + g);
        for ii = 2:M-1
            f(ii,:) = 0.5 * prod(x(1:M-ii,:), 1) .* (1 - x(M-ii+1,:)) .* (1 + g);
        end
        f(M,:) = 0.5 * (1 - x(1,:)) .* (1 + g);
        y = f;
    end
end

%% DTLZ2 函数生成器
% 建议参数：k = 10
function p = DTLZ2(p, dim)
    global M k;
    p.name = 'DTLZ2';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ2 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum((xm - 0.5).^2, 1);

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-ii,:)), 1) .* sin(pi/2 * x(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(pi/2 * x(1,:));
        y = f;
    end
end

%% DTLZ3 函数生成器
% 建议参数：k = 10
function p = DTLZ3(p, dim)
    global M k;
    p.name = 'DTLZ3';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ3 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = 100 * (k + sum((xm - 0.5).^2 - cos(20 * pi * (xm - 0.5)), 1));

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-ii,:)), 1) .* sin(pi/2 * x(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(pi/2 * x(1,:));
        y = f;
    end
end

%% DTLZ4 函数生成器
% 建议参数：k = 10
function p = DTLZ4(p, dim)
    global M k;
    p.name = 'DTLZ4';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ4 评价函数
    function y = evaluate(x)
        x = x';
        alpha = 100; % 形状参数
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum((xm - 0.5).^2, 1);

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-1,:).^alpha), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-ii,:).^alpha), 1) .* sin(pi/2 * x(M-ii+1,:).^alpha);
        end
        f(M,:) = (1 + g) .* sin(pi/2 * x(1,:).^alpha);
        y = f;
    end
end

%% DTLZ5 函数生成器
% 建议参数：k = 10
function p = DTLZ5(p, dim)
    global M k;
    p.name = 'DTLZ5';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ5 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum((xm - 0.5).^2, 1); 

        % 计算 theta
        theta = pi/2 * x(1,:);
        gr = g(ones(M-2,1), :); % 复制 g 以进行后续运算
        theta(2:M-1,:) = pi ./ (4 * (1 + gr)) .* (1 + 2 * gr .* x(2:M-1,:));

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(theta(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(theta(1:M-ii,:)), 1) .* sin(theta(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(theta(1,:));
        y = f;
    end
end

%% DTLZ6 函数生成器
% 建议参数：k = 10
function p = DTLZ6(p, dim)
    global M k;
    p.name = 'DTLZ6';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ6 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum(xm.^0.1, 1); 

        % 计算 theta
        theta = pi/2 * x(1,:);
        gr = g(ones(M-2,1), :); % 复制 g 以进行后续运算
        theta(2:M-1,:) = pi ./ (4 * (1 + gr)) .* (1 + 2 * gr .* x(2:M-1,:));

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(theta(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(theta(1:M-ii,:)), 1) .* sin(theta(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(theta(1,:));
        y = f;
    end
end

%% DTLZ7 函数生成器
% 建议参数：k = 20
function p = DTLZ7(p, dim)
    global M k;
    p.name = 'DTLZ7';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ7 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 20 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        % 计算 g 函数
        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = 1 + 9/k * sum(xm, 1);

        % 计算前 M-1 个目标函数
        f(1:M-1,:) = x(1:M-1,:);
        
        % 计算最后一个目标函数
        gaux = g(ones(M-1,1), :); % 复制 g 以进行后续运算
        h = M - sum(f ./ (1 + gaux) .* (1 + sin(3 * pi * f)), 1);
        f(M,:) = (1 + g) .* h;
        y = f;
    end
end

%% --------------WFG 基准测试--------参考文献：[1]----
% WFG系列问题定义

%% WFG1 函数生成器
% 决策变量维数：dim = k + l;
function p = wfg1(p, dim)
    global k l M;
    p.name = 'WFG1';
    p.pd = dim;        % 决策变量维数
    p.od = M;          % 目标函数维数
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG1 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 1);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：b_flat
        Ybarbar = Ybar;
        biasA = 0.8;
        biasB = 0.75;
        biasC = 0.85;
        Ybarbar(:, lLoop:n) = b_flat(Ybarbar(:, lLoop:n), biasA, biasB, biasC);
        
        % 应用第三变换：b_poly
        Ybarbarbar = Ybarbar;
        biasA = 0.02;
        Ybarbarbar = b_poly(Ybarbarbar, biasA);
        
        % 应用第四变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            weights = 2 * (lBnd:uBnd);
            T(:,i) = r_sum(Ybarbarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbarbar(:, lLoop:n), 2*(lLoop:n));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(1, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        fM = h_mixed(X(:,1), 1, 5);
        F = h_convex(X(:,1:uLoop));
        F(:,M) = fM;
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* F;
        y = F';
    end
end

%% WFG2 函数生成器
function p = wfg2(p, dim)
    global k l M;
    p.name = 'WFG2';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG2 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 2);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：r_nonsep
        Ybarbar = Ybar;
        uLoop = k + l/2;
        for i = lLoop:uLoop
            lBnd = k + 2*(i - k) - 1;
            uBnd = k + 2*(i - k);
            Ybarbar(:,i) = r_nonsep(Ybar(:, lBnd:uBnd), 2);
        end
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:k + l/2), ones(1, (k + l/2) - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(2, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        fM = h_disc(X(:,1), 1, 1, 5);
        F = h_convex(X(:,1:uLoop));
        F(:,M) = fM;
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* F;
        y = F';
    end
end

%% WFG3 函数生成器
function p = wfg3(p, dim)
    global k l M;
    p.name = 'WFG3';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG3 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 3);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：r_nonsep
        Ybarbar = Ybar;
        uLoop = k + l/2;
        for i = lLoop:uLoop
            lBnd = k + 2*(i - k) - 1;
            uBnd = k + 2*(i - k);
            Ybarbar(:,i) = r_nonsep(Ybar(:, lBnd:uBnd), 2);
        end
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:k + l/2), ones(1, (k + l/2) - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(3, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_linear(X(:,1:uLoop));
        y = F';
    end
end

%% WFG4 函数生成器
function p = wfg4(p, dim)
    global k l M;    
    p.name = 'WFG4';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG4 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 4);
        testNo = 5;

        % 应用第一变换：根据 testNo 选择不同的变换
        if testNo == 4
            shiftA = 30;
            shiftB = 10;
            shiftC = 0.35;
            Ybar = s_multi(Y, shiftA, shiftB, shiftC);
        else
            shiftA = 0.35;
            shiftB = 0.001;
            shiftC = 0.05;
            Ybar = s_decep(Y, shiftA, shiftB, shiftC);
        end
        
        % 应用第二变换：r_sum
        T = NaN * ones(noSols, M);
        lLoop = k + 1;
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(4, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG5 函数生成器
function p = wfg5(p, dim)
    global k l M;
    p.name = 'WFG5';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG5 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 5);

        % 应用第一变换：s_decep
        shiftA = 0.35;
        shiftB = 0.001;
        shiftC = 0.05;
        Ybar = s_decep(Y, shiftA, shiftB, shiftC);
        
        % 应用第二变换：r_sum
        T = NaN * ones(noSols, M);
        lLoop = k + 1;
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(5, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG6 函数生成器
function p = wfg6(p, dim)
    global k l M;
    p.name = 'WFG6';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG6 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 6);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：r_nonsep
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_nonsep(Ybar(:, lBnd:uBnd), k/(M-1));
        end
        T(:,M) = r_nonsep(Ybar(:, k+1:k+l), l);
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(6, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG7 函数生成器
function p = wfg7(p, dim)
    global k l M;
    p.name = 'WFG7';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG7 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 7);

        % 应用第一变换：b_param
        Ybar = Y;
        biasA = 0.98 / 49.98;
        biasB = 0.02;
        biasC = 50;
        for i = 1:k
            Ybar(:,i) = b_param(Y(:,i), r_sum(Y(:,i+1:n), ones(1, n-i)), biasA, biasB, biasC);
        end
        
        % 应用第二变换：s_decep 和 s_multi
        Ybarbar = Ybar;
        lLoop = k + 1;
        shiftA = 0.35;
        shiftB = 0.001;
        shiftC = 0.05;
        Ybarbar(:, lLoop:n) = s_decep(Ybar(:, lLoop:n), shiftA, shiftB, shiftC);
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(7, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG8 函数生成器
function p = wfg8(p, dim)
    global k l M;
    p.name = 'WFG8';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG8 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 8);

        % 应用第一变换：b_param
        Ybar = Y;
        lLoop = k + 1;
        biasA = 0.98 / 49.98;
        biasB = 0.02;
        biasC = 50;
        for i = lLoop:n
            Ybar(:,i) = b_param(Y(:,i), r_sum(Y(:,1:i-1), ones(1, i-1)), biasA, biasB, biasC);
        end
        
        % 应用第二变换：s_linear
        Ybarbar = Ybar;
        shiftA = 0.35;
        Ybarbar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(8, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG9 函数生成器
function p = wfg9(p, dim)
    global k l M;
    p.name = 'WFG';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG9 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 9);

        % 应用第一变换：b_param
        Ybar = Y;
        uLoop = n - 1;
        biasA = 0.98 / 49.98;
        biasB = 0.02;
        biasC = 50;
        for i = 1:uLoop
            Ybar(:,i) = b_param(Y(:,i), r_sum(Y(:,i+1:n), ones(1, n-i)), biasA, biasB, biasC);
        end
        
        % 应用第二变换：s_decep 和 s_multi
        Ybarbar = Ybar;
        shiftA = 0.35;
        shiftB = 0.001;
        shiftC = 0.05;
        Ybarbar(:,1:k) = s_decep(Ybar(:,1:k), shiftA, shiftB, shiftC);
        biasA = 30;
        biasB = 95;
        biasC = 0.35;
        Ybarbar(:,k+1:n) = s_multi(Ybar(:,k+1:n), shiftA, shiftB, shiftC);
        
        % 应用第三变换：r_nonsep
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_nonsep(Ybarbar(:, lBnd:uBnd), k/(M-1));
        end
        T(:,M) = r_nonsep(Ybarbar(:,k+1:k+l), l);
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(9, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG10 函数生成器
function p = wfg10(p, dim)
    global k l M;
    p.name = 'WFG10';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG10 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 10);

        % 应用第一变换：s_multi
        shiftA = 30;
        shiftB = 10;
        shiftC = 0.35;
        Ybar = s_multi(Y, shiftA, shiftB, shiftC);
        
        % 应用第二变换：r_sum
        T = NaN * ones(noSols, M);
        lLoop = k + 1;
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(10, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_convex(X(:,1:uLoop));
        y = F';
    end
end

%% ----------%% CEC2009 系列无约束。参考文献：[4]---------
% CEC2009 UF（Unconstrained Function）测试问题

%% UF1 函数生成器
% x和y为列向量，输入x必须在搜索空间内，可以是矩阵
function p = uf1(p, dim)
    p.name = 'uf1';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF1 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = (x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp1 = sum(tmp(3:2:dim,:));  % 奇数索引
        tmp2 = sum(tmp(2:2:dim,:));  % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp2 / size(2:2:dim,2);
        clear tmp;
    end
end

%% UF2 函数生成器
function p = uf2(p, dim)
    p.name = 'uf2';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0;
    p.func = @evaluate;
    
    % UF2 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        X1 = repmat(x(1,:), [dim-1,1]);
        A = 6 * pi * X1 + pi/dim * repmat((2:dim)', [1, num]);
        tmp = zeros(dim, num);    
        tmp(2:dim,:) = (x(2:dim,:) - 0.3 * X1 .* (X1 .* cos(4.0 * A) + 2.0) .* cos(A)).^2;
        tmp1 = sum(tmp(3:2:dim,:));  % 奇数索引
        tmp(2:dim,:) = (x(2:dim,:) - 0.3 * X1 .* (X1 .* cos(4.0 * A) + 2.0) .* sin(A)).^2;
        tmp2 = sum(tmp(2:2:dim,:));  % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp2 / size(2:2:dim,2);
        clear X1 A tmp;
    end
end

%% UF3 函数生成器
function p = uf3(p, dim)
    p.name = 'uf3';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % UF3 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - repmat(x(1,:), [dim-1,1]).^(0.5 + 1.5 * (repmat((2:dim)', [1, num]) - 2.0) / (dim - 2.0));
        tmp1 = sum(Y(2:dim,:).^2, 1);
        tmp2 = sum(cos(20.0 * pi * Y(2:dim,:) ./ sqrt(repmat((2:dim)', [1, num]))), 1);
        tmp11 = 4.0 * sum(tmp1(3:2:dim,:)) - 2.0 * prod(tmp2(3:2:dim,:)) + 2.0;  % 奇数索引
        tmp21 = 4.0 * sum(tmp1(2:2:dim,:)) - 2.0 * prod(tmp2(2:2:dim,:)) + 2.0;  % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp11 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp21 / size(2:2:dim,2);
        clear Y;
    end
end

%% UF4 函数生成器
function p = uf4(p, dim)
    p.name = 'uf4';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.domain(1,2) = 1; % 第一个决策变量上界
    p.func = @evaluate;
    
    % UF4 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(Y(3:2:dim,:).^2, 1);  % 奇数索引
        tmp2 = sum(Y(4:2:dim,:).^2, 1);  % 偶数索引
        index1 = Y(2,:) < (1.5 - 0.75 * sqrt(2.0));
        index2 = Y(2,:) >= (1.5 - 0.75 * sqrt(2.0));
        Y(2,index1) = abs(Y(2,index1));
        Y(2,index2) = 0.125 + (Y(2,index2) - 1.0).^2;
        y(1,:) = x(1,:) + tmp1;
        y(2,:) = (1.0 - x(1,:)).^2 + Y(2,:) + tmp2;
        t = x(2,:) - sin(6.0 * pi * x(1,:) + 2.0 * pi/dim) - 0.5 * x(1,:) + 0.25;
        y(1,:) = y(1,:);
        y(2,:) = y(2,:);
        c(1,:) = Y(1,:) + Y(2,:) - 1.0;
        clear Y;
    end
end

%% UF5 函数生成器
function p = uf5(p, dim)
    p.name = 'uf5';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF5 评价函数
    function y = evaluate(x)
        N = 10.0;
        E = 0.1;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        H = zeros(dim, num);
        H(2:dim,:) = 2.0 * Y(2:dim,:).^2 - cos(4.0 * pi * Y(2:dim,:)) + 1.0;
        tmp1 = sum(H(3:2:dim,:));  % 奇数索引
        tmp2 = sum(H(2:2:dim,:));  % 偶数索引
        tmp = (0.5/N + E) * abs(sin(2.0 * N * pi * x(1,:)));
        y(1,:) = x(1,:) + tmp + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + tmp + 2.0 * tmp2 / size(2:2:dim,2);
        clear Y H;
    end
end

%% UF6 函数生成器
function p = uf6(p, dim)
    p.name = 'uf6';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF6 评价函数
    function y = evaluate(x)
        N = 2.0;
        E = 0.1;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(Y(2:dim,:).^2, 1);
        tmp2 = sum(cos(20.0 * pi * Y(2:dim,:) ./ sqrt(repmat((2:dim)', [1, num]))), 1);
        tmp11 = 4.0 * sum(tmp1(3:2:dim,:)) - 2.0 * prod(tmp2(3:2:dim,:)) + 2.0;  % 奇数索引
        tmp21 = 4.0 * sum(tmp1(2:2:dim,:)) - 2.0 * prod(tmp2(2:2:dim,:)) + 2.0;  % 偶数索引
        tmp = max(0, (1.0/N + 2.0 * E) * sin(2.0 * N * pi * x(1,:)));
        y(1,:) = x(1,:) + tmp + 2.0 * tmp11 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + tmp + 2.0 * tmp21 / size(2:2:dim,2);
        clear Y tmp1 tmp2;
    end
end

%% UF7 函数生成器
function p = uf7(p, dim)
    p.name = 'uf7';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF7 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = (x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp1 = sum(Y(3:2:dim,:));  % 奇数索引
        tmp2 = sum(Y(2:2:dim,:));  % 偶数索引
        tmp = (x(1,:)).^0.2;
        y(1,:) = tmp + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - tmp + 2.0 * tmp2 / size(2:2:dim,2);
        clear Y;
    end
end

%% UF8 函数生成器
function p = uf8(p, dim)
    p.name = 'uf8';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % UF8 评价函数
    function y = evaluate(x)
        N = 2.0;
        a = 4.0;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * abs(sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0))) - 1.0;
        clear Y;
    end
end

%% UF9 函数生成器
function p = uf9(p, dim)
    p.name = 'uf9';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % UF9 评价函数
    function y = evaluate(x)
        N = 2.0;
        a = 3.0;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y;
    end
end

%% UF10 函数生成器
function p = uf10(p, dim)
    p.name = 'uf10';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % UF10 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]));
        H = zeros(dim, num);
        H(3:dim,:) = 4.0 * Y(3:dim,:).^2 - cos(8.0 * pi * Y(3:dim,:)) + 1.0;
        tmp1 = sum(H(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(H(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(H(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y H;
    end
end

%% ----------%% CEC2009 系列有约束。参考文献：[4]---------
% CEC2009 CF（Constrained Function）测试问题

%% CF1 函数生成器
% x和y为列向量，输入x必须在搜索空间内，可以是矩阵
function p = cf1(p, dim)
    p.name = 'cf1';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % CF1 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 10.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = (x(2:dim,:) - repmat(x(1,:), [dim-1,1]).^(0.5 + 1.5 * (repmat((2:dim)', [1, num]) - 2.0) / (dim - 2.0))).^2;
        tmp1 = sum(Y(3:2:dim,:));    % 奇数索引
        tmp2 = sum(Y(2:2:dim,:));    % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + 2.0 * tmp2 / size(2:2:dim,2);
        c(1,:) = y(1,:) + y(2,:) - a * abs(sin(N * pi * (y(1,:)-y(2,:)+1.0))) - 1.0;
        clear Y;
    end
end

%% CF2 函数生成器
function p = cf2(p, dim)
    p.name = 'cf2';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0;
    p.func = @evaluate;
    
    % CF2 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 2.0;
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = (x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp1 = sum(tmp(3:2:dim,:));    % 奇数索引
        tmp(2:dim,:) = (x(2:dim,:) - cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp2 = sum(tmp(2:2:dim,:));    % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp2 / size(2:2:dim,2);
        t = y(2,:) + sqrt(y(1,:)) - a * sin(N * pi * (sqrt(y(1,:)) - y(2,:) + 1.0)) - 1.0;
        c(1,:) = sign(t) .* abs(t) ./ (1.0 + exp(4.0 * abs(t)));
        clear tmp;
    end
end

%% CF3 函数生成器
function p = cf3(p, dim)
    p.name = 'cf3';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.func = @evaluate;
    
    % CF3 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 2.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(2.0 * Y(3:2:dim,:).^2 - cos(4.0 * pi * Y(3:2:dim,:)) + 1.0, 1);  % 奇数索引
        tmp2 = sum(2.0 * Y(4:2:dim,:).^2 - cos(4.0 * pi * Y(4:2:dim,:)) + 1.0, 1);  % 偶数索引
        tmp = 0.5 * (1 - x(1,:)) - (1 - x(1,:)).^2;
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:).^2 + 2.0 * tmp2 / size(2:2:dim,2);
        c(1,:) = y(2,:) + y(1,:).^2 - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y;
    end
end

%% CF4 函数生成器
function p = cf4(p, dim)
    p.name = 'cf4';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF4 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(tmp(3:2:dim,:).^2, 1);  % 奇数索引
        tmp2 = sum(tmp(4:2:dim,:).^2, 1);  % 偶数索引
        index1 = tmp(2,:) < (1.5 - 0.75 * sqrt(2.0));
        index2 = tmp(2,:) >= (1.5 - 0.75 * sqrt(2.0));
        tmp(2,index1) = abs(tmp(2,index1));
        tmp(2,index2) = 0.125 + (tmp(2,index2) - 1.0).^2;
        y(1,:) = x(1,:) + tmp1;
        y(2,:) = (1.0 - x(1,:)).^2 + tmp(2,:) + tmp2;
        t = x(2,:) - sin(6.0 * pi * x(1,:) + 2.0 * pi/dim) - 0.5 * x(1,:) + 0.25;
        c(1,:) = sign(t) .* abs(t) ./ (1.0 + exp(4.0 * abs(t)));
        clear tmp index1 index2;
    end
end

%% CF5 函数生成器
function p = cf5(p, dim)
    p.name = 'cf5';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF5 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - 0.8 * repmat(x(1,:), [dim-1,1]) .* cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(2.0 * tmp(3:2:dim,:).^2 - cos(4.0 * pi * tmp(3:2:dim,:)) + 1.0, 1);  % 奇数索引
        tmp(2:dim,:) = x(2:dim,:) - 0.8 * repmat(x(1,:), [dim-1,1]) .* sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));    
        tmp2 = sum(2.0 * tmp(4:2:dim,:).^2 - cos(4.0 * pi * tmp(4:2:dim,:)) + 1.0, 1);  % 偶数索引
        index1 = tmp(2,:) < (1.5 - 0.75 * sqrt(2.0));
        index2 = tmp(2,:) >= (1.5 - 0.75 * sqrt(2.0));
        tmp(2,index1) = abs(tmp(2,index1));
        tmp(2,index2) = 0.125 + (tmp(2,index2) - 1.0).^2;
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + tmp(2,:) + 2.0 * tmp2 / size(2:2:dim,2);
        c(1,:) = x(2,:) - 0.8 * x(1,:) .* sin(6.0 * pi * x(1,:) + 2.0 * pi/dim) - 0.5 * x(1,:) + 0.25;
        clear tmp;
    end
end

%% CF6 函数生成器
function p = cf6(p, dim)
    p.name = 'cf6';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF6 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - 0.8 * repmat(x(1,:), [dim-1,1]) .* cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(tmp(3:2:dim,:).^2, 1);  % 奇数索引
        tmp2 = sum(tmp(2:2:dim,:).^2, 1);  % 偶数索引
        y(1,:) = x(1,:) + tmp1;
        y(2,:) = (1.0 - x(1,:)).^2 + tmp2;
        tmp = 0.5 * (1 - x(1,:)) - (1 - x(1,:)).^2;
        c(1,:) = x(2,:) - 0.8 * x(1,:) .* sin(6.0 * pi * x(1,:) + 2 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));
        tmp = 0.25 * sqrt(1 - x(1,:)) - 0.5 * (1 - x(1,:));
        c(2,:) = x(4,:) - 0.8 * x(1,:) .* sin(6.0 * pi * x(1,:) + 4 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));    
        clear tmp;
    end
end

%% CF7 函数生成器
function p = cf7(p, dim)
    p.name = 'cf7';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF7 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(2.0 * tmp(3:2:dim,:).^2 - cos(4.0 * pi * tmp(3:2:dim,:)) + 1.0, 1);  % 奇数索引
        tmp(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp2 = sum(2.0 * tmp(6:2:dim,:).^2 - cos(4.0 * pi * tmp(6:2:dim,:)) + 1.0, 1);  % 偶数索引
        tmp(2,:) = tmp(2,:).^2;
        tmp(4,:) = tmp(4,:).^2;
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = (1.0 - x(1,:)).^2 + tmp(2,:) + tmp(4,:) + tmp2;
        tmp = 0.5 * (1 - x(1,:)) - (1 - x(1,:)).^2;
        c(1,:) = x(2,:) - sin(6.0 * pi * x(1,:) + 2 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));
        tmp = 0.25 * sqrt(1 - x(1,:)) - 0.5 * (1 - x(1,:));
        c(2,:) = x(4,:) - sin(6.0 * pi * x(1,:) + 4 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));    
        clear tmp;
    end
end

%% CF8 函数生成器
function p = cf8(p, dim)
    p.name = 'cf8';
    p.pd = dim;
    p.od = 3;
    p.domain = [-4 * ones(dim,1) 4 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF8 评价函数
    function [y, c] = evaluate(x)
        x = x';
        N = 2.0;
        a = 4.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * abs(sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0))) - 1.0;
        clear Y;
    end
end

%% CF9 函数生成器
function p = cf9(p, dim)
    p.name = 'cf9';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF9 评价函数
    function [y, c] = evaluate(x)
        x = x';
        N = 2.0;
        a = 3.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y;
    end
end

%% CF10 函数生成器
function p = cf10(p, dim)
    p.name = 'cf10';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF10 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 2.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        H = zeros(dim, num);
        H(3:dim,:) = 4.0 * Y(3:dim,:).^2 - cos(8.0 * pi * Y(3:dim,:)) + 1.0;
        tmp1 = sum(H(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(H(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(H(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y H;
    end
end

%% ------------------Transformation functions------------------
% ---------注意：以下函数不是测试问题，仅用于变换操作----------------

% WFG基准测试初始化
function [noSols, n, S, D, A, Y] = wfg_initialize(Z, M, k, l, testNo)
    % 检查输入参数数量
    if nargin ~= 5
        error('需要五个输入参数。');
    end

    % 获取决策变量数量和候选解数量
    [noSols, n] = size(Z);

    % 数据输入检查
    if n ~= (k + l)
        error('决策变量数量与k + l不一致。');
    end
    if rem(k, M-1) ~= 0
        error('k必须能被M-1整除。');
    end
    if (testNo == 2 || testNo == 3) && (rem(l,2) ~= 0)
        error('对于WFG2和WFG3，l必须是2的倍数。');
    end

    % 初始化函数范围内的常数
    NO_TESTS = 10;
    S = NaN * ones(1, M);
    for i = 1:M
        S(i) = 2 * i;
    end
    D = 1;
    A = ones(NO_TESTS, M-1);
    A(3,2:M-1) = 0;

    % 将所有变量范围转换到 [0,1]
    x = Z;
    for i = 1:n
        Y(:,i) = Z(:,i) ./ (2 * i);
    end
end

% Reduction: 加权和
function ybar = r_sum(y, weights)
    [noSols, noY] = size(y);
    wgtMatrix = repmat(weights, [noSols, 1]);
    ybar = y .* wgtMatrix;
    ybar = sum(ybar, 2) ./ sum(wgtMatrix, 2);
end

% Reduction: 非可分
function y_bar = r_nonsep(y, A)
    [noSols, noY] = size(y);
    y_bar = 0;
    for j = 1:noY
        innerSum = 0;
        for k = 0:(A-2)
            innerSum = innerSum + abs(y(:,j) - y(:,1 + mod(j + k, noY)));
        end
        y_bar = y_bar + y(:,j) + innerSum;
    end
    y_bar = y_bar / ((noY/A) * ceil(A/2) * (1 + 2*A - 2*ceil(A/2)));
end

% Bias: 多项式
function y_bar = b_poly(y, alpha)
    y_bar = y.^alpha;
end

% Bias: 平坦区域
function y_bar = b_flat(y, A, B, C)
    [noSols, noY] = size(y);
    min1 = min(0, floor(y - B));
    min2 = min(0, floor(C - y));
    y_bar = A + min1 * A .* (B - y) / B - min2 * (1 - A) .* (y - C) / (1 - C);
    % 由于机器精度问题，强制y_bar >= 0
    y_bar = max(0, y_bar);
end

% Bias: 参数依赖
function ybar = b_param(y, uy, A, B, C)
    [noSols, noY] = size(y);
    v = A - (1 - 2 * uy) .* abs(floor(0.5 - uy) + A);
    v = repmat(v, [1 noY]);
    ybar = y.^(B + (C - B) * v);
end

% Shift: 线性
function ybar = s_linear(y, A)
    ybar = abs(y - A) ./ abs(floor(A - y) + A);
end

% Shift: 欺骗性
function ybar = s_decep(y, A, B, C)
    y1 = floor(y - A + B) * (1 - C + (A - B)/B) / (A - B);
    y2 = floor(A + B - y) * (1 - C + (1 - A - B)/B) / (1 - A - B);
    ybar = 1 + (abs(y - A) - B) .* (y1 + y2 + 1/B);
end

% Shift: 多模态
function ybar = s_multi(y, A, B, C)
    y1 = abs(y - C) ./ (2 * (floor(C - y) + C));
    ybar = (1 + cos((4 * A + 2) * pi * (0.5 - y1)) + 4 * B * y1.^2) / (B + 2);
end

% Shape函数：线性
function f = h_linear(x)
    [noSols, mMinusOne] = size(x);
    M = mMinusOne + 1;
    f = NaN * ones(noSols, M);
    
    f(:,1) = prod(x, 2);
    for i = 2:mMinusOne
        f(:,i) = prod(x(:,1:M-i), 2) .* (1 - x(:,M-i+1));
    end
    f(:,M) = 1 - x(:,1);
end

% Shape函数：凸形
function f = h_convex(x)
    [noSols, mMinusOne] = size(x);
    M = mMinusOne + 1;
    f = NaN * ones(noSols, M);
    
    f(:,1) = prod(1 - cos(x * pi / 2), 2);
    for i = 2:mMinusOne
        f(:,i) = prod(1 - cos(x(:,1:M-i) * pi / 2), 2) .* (1 - sin(x(:,M-i+1) * pi / 2));
    end
    f(:,M) = 1 - sin(x(:,1) * pi / 2);
end

% Shape函数：凹形
function f = h_concave(x)
    [noSols, mMinusOne] = size(x);
    M = mMinusOne + 1;
    f = NaN * ones(noSols, M);
    
    f(:,1) = prod(sin(x * pi / 2), 2);
    for i = 2:mMinusOne
        f(:,i) = prod(sin(x(:,1:M-i) * pi / 2), 2) .* cos(x(:,M-i+1) * pi / 2);
    end
    f(:,M) = cos(x(:,1) * pi / 2);
end

% Shape函数：混合
function f = h_mixed(x, alpha, A)
    f = (1 - x(:,1) - cos(2 * A * pi * x(:,1) + pi/2) / (2 * A * pi)).^alpha;
end

% Shape函数：离散
function f = h_disc(x, alpha, beta, A)
    f = 1 - x(:,1).^alpha .* cos(A * x(:,1).^beta * pi).^2;
end

% 矩阵复制函数
function MatOut = rep(MatIn, REPN)
    % 获取输入矩阵的大小
    [N_D, N_L] = size(MatIn);
    
    % 计算复制索引
    Ind_D = rem(0:REPN(1)*N_D-1, N_D) + 1;
    Ind_L = rem(0:REPN(2)*N_L-1, N_L) + 1;
    
    % 创建输出矩阵
    MatOut = MatIn(Ind_D, Ind_L);
end
```

---

### **详细说明**

#### 1. **函数功能概述**

- **`testmop` 函数**：
  - 该函数用于生成指定名称和维数的多目标优化测试问题。
  - 通过动态调用相应的测试问题生成器函数（如 `zdt1`, `dtlz2`, `wfg1`, `uf1`, `cf1` 等），返回包含测试问题详细信息的结构体 `mop`。

- **各个测试问题生成器函数**：
  - **ZDT系列**：适用于二维目标问题，常用于验证多目标优化算法的性能。
  - **DTLZ系列**：适用于多维目标问题，具有可扩展性，适合测试算法在高维目标空间的表现。
  - **WFG系列**：具有多种特性（如可分离性、非可分离性、多模态性等），适合测试算法在不同问题特性下的适应性。
  - **CEC2009 UF和CF系列**：复杂的有约束和无约束的多目标优化问题，常用于评估算法的实际应用能力。

#### 2. **主要步骤解析**

1. **初始化阶段**：
   - **输入参数**：
     - `testname`：指定的测试问题名称。
     - `dimension`：决策变量的维数。
   - **生成测试问题结构体**：
     - 初始化一个空的结构体 `mop`，包含字段 `name`、`od`、`pd`、`domain` 和 `func`。
     - 使用 `eval` 函数动态调用对应的测试问题生成器函数，根据 `testname` 和 `dimension` 生成完整的 `mop` 结构体。

2. **各个测试问题生成器函数**：
   - **ZDT系列（`zdt1` 至 `zdt6`）**：
     - 每个函数设置特定的决策变量维数、目标函数维数、决策变量范围以及目标函数的计算方式。
     - 例如，`zdt1` 定义了一个具有两个目标函数的测试问题，其目标函数依赖于前一个决策变量和一个辅助函数 `g`。
   
   - **DTLZ系列（`DTLZ1` 至 `DTLZ7`）**：
     - 每个函数设置特定的决策变量维数、目标函数维数、决策变量范围以及目标函数的计算方式。
     - DTLZ问题通常通过辅助函数 `g` 来控制多目标之间的关系和多样性。
   
   - **WFG系列（`wfg1` 至 `wfg10`）**：
     - 每个函数设置特定的决策变量维数、目标函数维数、决策变量范围以及目标函数的计算方式。
     - WFG问题通常涉及一系列的变换函数（如 `s_linear`, `b_flat`, `b_poly`, `r_sum`, `r_nonsep` 等）来增加问题的复杂性。
   
   - **CEC2009 UF系列（`uf1` 至 `uf10`）**：
     - 定义了一系列无约束的多目标优化问题。
     - 每个函数设置特定的决策变量维数、目标函数维数、决策变量范围以及目标函数的计算方式。
   
   - **CEC2009 CF系列（`cf1` 至 `cf10`）**：
     - 定义了一系列有约束的多目标优化问题。
     - 每个函数设置特定的决策变量维数、目标函数维数、决策变量范围、目标函数的计算方式以及约束条件。

3. **变换函数**：
   - 这些函数用于WFG系列问题的预处理和变换，增强问题的复杂性和多样性。
   - 包括减缩函数（如 `r_sum`, `r_nonsep`）、偏置函数（如 `b_poly`, `b_flat`, `b_param`）、位移函数（如 `s_linear`, `s_decep`, `s_multi`）以及形状函数（如 `h_linear`, `h_convex`, `h_concave`, `h_mixed`, `h_disc`）。
   - 还有一个辅助的矩阵复制函数 `rep`，用于复制和扩展矩阵。

#### 3. **关键算法步骤解析**

- **动态调用测试问题生成器**：
  - 通过 `eval` 函数，根据输入的 `testname` 动态调用相应的测试问题生成器函数。例如，如果 `testname` 为 `'zdt1'`，则调用 `zdt1(p, dim)` 来生成ZDT1问题。

- **测试问题生成器函数**：
  - **结构体字段设置**：
    - `name`：设置测试问题的名称。
    - `pd`：设置决策变量的维数。
    - `od`：设置目标函数的维数。
    - `domain`：设置决策变量的边界范围，通常为 `[lower_bound, upper_bound]`。
    - `func`：设置目标函数的句柄，用于后续的评价和优化过程。

  - **目标函数计算**：
    - 每个测试问题都有其独特的目标函数计算方式，通常基于输入的决策变量 `x` 计算多个目标函数值 `y`。
    - 有些测试问题涉及辅助函数或参数（如 `g` 函数、`theta` 变量等）来控制目标函数之间的关系和多样性。

- **变换函数**：
  - **减缩函数**：
    - `r_sum`：计算加权和，常用于将多个变量合并为一个变量。
    - `r_nonsep`：计算非可分的减缩和，增加变量之间的相互依赖性。
  
  - **偏置函数**：
    - `b_poly`：多项式偏置，常用于调整变量的分布。
    - `b_flat`：平坦区域偏置，常用于创建扁平的目标函数区域。
    - `b_param`：参数依赖偏置，根据输入参数调整偏置程度。
  
  - **位移函数**：
    - `s_linear`：线性位移，常用于调整变量的位置。
    - `s_decep`：欺骗性位移，增加目标函数的复杂性。
    - `s_multi`：多模态位移，增加目标函数的多峰特性。
  
  - **形状函数**：
    - `h_linear`：线性形状函数，适用于线性目标函数。
    - `h_convex`：凸形形状函数，适用于凸优化问题。
    - `h_concave`：凹形形状函数，适用于凹优化问题。
    - `h_mixed`：混合形状函数，结合多种形状特性。
    - `h_disc`：离散形状函数，适用于离散优化问题。
  
  - **辅助函数**：
    - `rep`：矩阵复制函数，用于复制和扩展矩阵，以适应不同的运算需求。

#### 4. **注意事项**

- **全局变量使用**：
  - 部分测试问题（如 DTLZ 和 WFG 系列）依赖于全局变量 `M`（目标函数数量）和 `k`（控制维数的参数），在调用这些测试问题之前，必须正确设置这些全局变量。
  - 例如：
    ```matlab
    global M k;
    M = 3; % 设置目标函数数量
    k = 10; % 设置控制维数的参数
    mop = testmop('dtlz2', M-1 + k);
    ```

- **决策变量范围**：
  - 不同的测试问题有不同的决策变量范围，例如ZDT系列通常在 `[0,1]^dim`，而WFG系列在 `[0, 2i]^dim`。
  - 在定义决策变量范围时，需确保符合测试问题的要求。

- **约束条件**：
  - 对于有约束的测试问题（如CF系列），目标函数计算时会同时返回约束条件的违背程度 `c`。
  - 约束条件通常以 `c <= 0` 表示，算法在优化过程中需要考虑这些约束。

- **函数输入输出格式**：
  - 目标函数的输入 `x` 和输出 `y` 通常为列向量，可以是矩阵形式。
  - 对于有约束的测试问题，输出还包括约束条件 `c`。

- **变换函数的使用**：
  - 变换函数在WFG系列问题中广泛使用，用于增加问题的复杂性和多样性。
  - 这些函数需要按照特定的顺序和参数调用，以确保正确地生成目标函数值。

- **性能优化**：
  - 由于部分测试问题涉及大量的矩阵运算和循环，可能会对计算性能产生影响。
  - 可以考虑优化循环结构或使用向量化操作来提高运行效率。

#### 5. **参考文献**

- [1] WFG系列测试问题的相关文献。
- [2] Deb, K., "A Fast Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182-197, 2002.
- [3] Deb, K., "Scalable Multi-objective Optimization Test Problems," *Evolutionary Computation*, vol. 8, no. 2, pp. 189-199, 2000.
- [4] CEC2009 约束多目标优化测试问题相关文献。

---

### **总结**

`testmop.m` 文件实现了多种多目标优化测试问题的生成，包括常见的ZDT、DTLZ、WFG系列问题以及CEC2009的UF和CF系列问题。通过输入测试问题的名称和决策变量的维数，可以生成相应的测试问题结构体，包含目标函数、决策变量范围等信息。这些测试问题广泛应用于评估和验证多目标优化算法的性能和适应性。
