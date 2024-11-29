# 基于范围选择的进化多目标优化PESA-II概述

**PESA-II（Pareto Envelope-based Selection Algorithm II）**是一种基于网格（Grid）的进化多目标优化算法，旨在有效地搜索和维护Pareto前沿（Pareto Front）上的非支配解。PESA-II在处理多目标优化问题时，通过引入网格结构来增强解集的多样性和覆盖性，从而提高算法的整体性能。

#### 一、PESA-II的基本概念

1. **多目标优化（Multi-Objective Optimization, MOO）**：在实际问题中，往往需要同时优化多个相互冲突的目标函数。例如，在设计飞机时，需要平衡重量和成本。在这种情况下，优化算法需要寻找多个折衷解，形成一个Pareto前沿。

2. **Pareto最优解（Pareto Optimal Solutions）**：在多目标优化中，Pareto最优解是指不存在其他解在所有目标上都优于该解。换句话说，任何改进一个目标的同时，至少会导致另一个目标的恶化。

3. **支配关系（Dominance Relationship）**：在PESA-II中，个体之间的支配关系用于判断解的优劣。具体来说，个体A支配个体B，若A在所有目标上不劣于B，并且至少在一个目标上优于B。

4. **网格划分（Grid Division）**：PESA-II通过将目标空间划分为多个网格单元，每个网格单元负责维护目标空间的一个区域。这种方法有助于保持解集的均匀分布，避免解集过于集中或稀疏。

#### 二、PESA-II算法步骤

PESA-II的核心步骤可以概括为以下几个部分：

1. **初始化**：
   - 随机生成初始种群（Population）。
   - 计算每个个体的目标值（Cost）。
   - 初始化存档（Archive），用于存储非支配解。

2. **支配关系判断**：
   - 通过`DetermineDomination.m`和`Dominates.m`函数，确定种群中每个个体是否被其他个体支配。

3. **存档管理**：
   - 将非支配解添加到存档中。
   - 如果存档大小超过预定限制，通过`TruncatePopulation.m`函数进行截断，删除过多的个体以保持存档大小。

4. **网格创建与分配**：
   - 通过`CreateGrid.m`函数，创建目标空间的网格结构，并将存档中的个体分配到相应的网格中。
   - 网格的上下界通过膨胀因子（InflationFactor）进行调整，确保覆盖更多的解空间。

5. **选择操作**：
   - 利用`SelectFromPopulation.m`和`RouletteWheelSelection.m`函数，根据网格中的个体数量和参数`beta`，选择个体进行交叉和变异操作。
   - `beta`参数控制选择概率与网格稀疏程度之间的关系，`beta`越大，稀疏网格的优先级越高。

6. **交叉与变异**：
   - 通过`Crossover.m`函数，执行交叉操作，生成新的子代个体。
   - 通过`Mutate.m`函数，执行变异操作，生成新的子代个体。

7. **种群更新**：
   - 将交叉和变异生成的子代个体添加到种群中，为下一次迭代做准备。

8. **迭代终止与结果展示**：
   - 重复上述步骤，直到达到最大迭代次数（MaxIt）。
   - 通过`PlotCosts.m`函数，绘制Pareto前沿，展示优化结果。

#### 三、结合代码分析PESA-II的实现

以下将结合您提供的MATLAB代码，详细解析PESA-II的各个组成部分及其实现细节。

##### 1. **网格创建与分配（CreateGrid.m）**

```matlab
function [pop, grid]=CreateGrid(pop,nGrid,InflationFactor)
% CreateGrid 创建网格并将个体分配到相应的网格中
% 输入参数:
%   pop - 当前种群
%   nGrid - 每个目标维度的网格数量
%   InflationFactor - 网格膨胀因子
% 输出参数:
%   pop - 更新后的种群（包含网格索引）
%   grid - 网格结构数组
```

- **功能**：将当前种群中的个体根据其目标值分配到预先定义的网格中。通过网格划分，PESA-II能够在目标空间中保持解集的多样性。
- **步骤**：
  - 计算每个目标的最小值和最大值，并根据膨胀因子调整网格范围。
  - 使用`linspace`函数在每个目标维度上生成网格分割点，包括负无穷大和正无穷大作为边界。
  - 初始化网格结构，记录每个网格的上下界、索引、成员数量和成员列表。
  - 调用`FindPositionInGrid`函数，将个体分配到对应的网格中。

##### 2. **支配关系判断（DetermineDomination.m 和 Dominates.m）**

```matlab
function pop=DetermineDomination(pop)
% DetermineDomination 确定种群中每个个体是否被支配
% 输入参数:
%   pop - 当前种群，包含目标值
% 输出参数:
%   pop - 更新后的种群，包含IsDominated标志
```

```matlab
function b=Dominates(x, y)
% Dominates 判断个体x是否支配个体y
% 支配条件: x在所有目标上都不劣于y，且至少在一个目标上优于y
% 输入参数:
%   x, y - 两个待比较的个体，包含Cost字段
% 输出参数:
%   b - 布尔值，若x支配y，则为true，否则为false
```

- **功能**：
  - `DetermineDomination.m`：通过遍历种群中的所有个体，利用`Dominates.m`函数判断每个个体是否被其他个体支配，并更新其`IsDominated`标志。
  - `Dominates.m`：具体判断两个个体之间的支配关系，若`x`在所有目标上不劣于`y`，且至少在一个目标上优于`y`，则认为`x`支配`y`。

##### 3. **网格分配（FindPositionInGrid.m）**

```matlab
function [pop, grid]=FindPositionInGrid(pop, grid)
% FindPositionInGrid 将种群中的个体分配到对应的网格中
% 输入参数:
%   pop - 当前种群，包含Cost字段
%   grid - 网格结构数组，包含LB和UB
% 输出参数:
%   pop - 更新后的种群，包含GridIndex
%   grid - 更新后的网格，包含成员信息
```

- **功能**：根据个体的目标值，将其分配到相应的网格中，并更新网格的成员信息。
- **步骤**：
  - 提取网格的下界（LB）和上界（UB）。
  - 初始化每个网格的成员数量和成员列表。
  - 对种群中的每个个体，调用`FindGridIndex`函数确定其所属网格，并更新种群和网格的信息。

##### 4. **选择操作（SelectFromPopulation.m 和 RouletteWheelSelection.m）**

```matlab
function P = SelectFromPopulation(pop, grid, beta)
% SelectFromPopulation 从种群中选择一个个体，基于网格和选择参数beta
% 输入参数:
%   pop - 当前种群
%   grid - 网格结构数组
%   beta - 选择操作的参数
% 输出参数:
%   P - 被选择的个体
```

```matlab
function i = RouletteWheelSelection(p)
% RouletteWheelSelection 使用轮盘赌选择法选择个体
% 输入参数:
%   p - 每个个体的选择概率向量
% 输出参数:
%   i - 被选择的个体的索引
```

- **功能**：
  - `SelectFromPopulation.m`：基于网格中的个体数量和参数`beta`，计算每个网格的选择概率，并通过轮盘赌方法选择一个个体。`beta`参数用于控制稀疏网格的优先选择。
  - `RouletteWheelSelection.m`：实现轮盘赌选择方法，根据概率向量`p`选择一个个体的索引。

##### 5. **截断操作（TruncatePopulation.m）**

```matlab
function [pop, grid] = TruncatePopulation(pop, grid, E, beta)
% TruncatePopulation 截断种群以满足存档大小限制
% 输入参数:
%   pop - 当前存档
%   grid - 网格结构数组
%   E - 需要删除的个体数量
%   beta - 删除操作的参数
% 输出参数:
%   pop - 更新后的存档
%   grid - 更新后的网格
```

- **功能**：当存档中的个体数量超过预定限制时，通过基于网格的删除策略，移除多余的个体以保持存档大小。
- **步骤**：
  - 循环执行`E`次删除操作，每次选择一个网格并从中随机删除一个个体。
  - 删除的网格选择概率基于网格中个体数量的`beta`次方，使得个体数量多的网格更可能被选择。
  - 更新网格的成员列表和个体数量，并记录待删除的个体索引。

##### 6. **交叉与变异操作（Crossover.m 和 Mutate.m）**

```matlab
function [y1, y2]=Crossover(x1, x2, params)
% Crossover 执行交叉操作，生成两个新的个体
% 输入参数:
%   x1, x2 - 父代个体的位置向量
%   params - 交叉操作的参数结构体，包含gamma, VarMin, VarMax
% 输出参数:
%   y1, y2 - 生成的子代个体的位置向量
```

```matlab
function y = Mutate(x, params)
% Mutate 对个体x执行变异操作
% 输入参数:
%   x - 原始个体的位置向量
%   params - 变异操作的参数结构体，包含h, VarMin, VarMax
% 输出参数:
%   y - 变异后的个体的位置向量
```

- **功能**：
  - `Crossover.m`：实现交叉操作，通过线性组合父代个体的位置向量，生成两个新的子代个体。交叉系数`alpha`在`[-gamma, 1 + gamma]`范围内随机生成，以控制子代的生成范围。
  - `Mutate.m`：实现变异操作，通过向个体的位置向量添加服从正态分布的随机噪声，引入新的基因多样性。变异幅度由参数`h`和决策变量的范围决定，确保变异后的个体仍在允许范围内。

##### 7. **主程序（pesa2.m）**

```matlab
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
```

- **功能**：`pesa2.m`是PESA-II算法的主程序，负责整个优化过程的执行，包括初始化、迭代循环、支配关系判断、存档管理、网格创建、选择、交叉与变异操作，以及结果的展示。

- **主要步骤**：
  1. **初始化**：
     - 清理工作环境。
     - 定义目标函数（`MOP2`）、决策变量的数量和范围。
     - 设置PESA-II的各种参数，如最大迭代次数、种群大小、存档大小、网格数量、膨胀因子、选择和删除参数、交叉和变异概率及其相关参数。
     - 初始化种群，随机生成个体的位置，并计算其目标值。
     - 初始化存档为空。

  2. **主循环**（迭代执行`MaxIt`次）：
     - **支配关系判断**：调用`DetermineDomination.m`函数，标记种群中每个个体是否被支配。
     - **非支配个体提取**：提取未被支配的个体，并将其添加到存档中。
     - **存档支配关系判断**：更新存档中每个个体的支配关系，保留非支配的存档个体作为当前的Pareto前沿。
     - **网格创建与分配**：调用`CreateGrid.m`函数，创建网格并将存档中的个体分配到对应的网格中。
     - **存档截断**：若存档大小超过`nArchive`，调用`TruncatePopulation.m`函数进行截断，删除多余的个体，并重新创建网格。
     - **Pareto前沿绘制**：使用`PlotCosts.m`函数绘制当前的Pareto前沿，实时观察优化进展。
     - **迭代信息显示**：在命令窗口输出当前迭代次数及Pareto前沿的成员数量。
     - **交叉与变异操作**：
       - **交叉**：根据交叉概率和次数，选择父代个体进行交叉，生成子代个体，并计算其目标值。
       - **变异**：根据变异概率和次数，选择父代个体进行变异，生成子代个体，并计算其目标值。
     - **种群更新**：将交叉和变异生成的子代添加到种群中，为下一次迭代做准备。

  3. **结果展示**：
     - 提取并显示Pareto前沿各目标的统计信息，包括最小值、最大值、范围、标准差和均值。
     - 通过绘图窗口展示最终的Pareto前沿分布。

##### 8. **目标函数（MOP2.m）**

```matlab
function z = MOP2(x)
% MOP2 定义了一个多目标优化问题（示例）
% 输入参数:
%   x - 决策变量向量
% 输出参数:
%   z - 目标值向量

    n = numel(x);  % 决策变量的数量
    
    % 计算第一个目标
    z1 = 1 - exp(-sum((x - 1 / sqrt(n)).^2));
    
    % 计算第二个目标
    z2 = 1 - exp(-sum((x + 1 / sqrt(n)).^2));
    
    % 返回目标值向量
    z = [z1
         z2];
    
end
```

- **功能**：定义了一个简单的双目标优化问题。第一个目标旨在最小化个体与点`1/sqrt(n)`的距离，第二个目标旨在最小化个体与点`-1/sqrt(n)`的距离。这两个目标相互冲突，迫使优化算法在它们之间寻找折衷解。

##### 9. **结果绘制（PlotCosts.m）**

```matlab
function PlotCosts(PF)
% PlotCosts 绘制Pareto前沿的目标值
% 输入参数:
%   PF - Pareto前沿的个体数组，包含Cost字段

    % 提取所有个体的目标值
    PFC = [PF.Cost];
    
    % 绘制第一个目标与第二个目标的关系
    plot(PFC(1, :), PFC(2, :), 'x');
    xlabel('1^{st} Objective');  % x轴标签
    ylabel('2^{nd} Objective');  % y轴标签
    grid on;                     % 显示网格

end
```

- **功能**：实时绘制Pareto前沿的目标值，帮助用户可视化优化过程中的解集分布。适用于双目标优化问题，显示第一个目标与第二个目标的关系。

#### 四、PESA-II的优势与局限

**优势**：

1. **多样性维护**：通过网格划分和基于范围的选择策略，PESA-II能够有效地维护解集的多样性，避免解集过于集中或稀疏。
2. **高效的存档管理**：利用存档结构，PESA-II能够动态地存储和管理非支配解，确保Pareto前沿的覆盖性和代表性。
3. **可扩展性**：PESA-II的网格划分方法可以适应不同维度的目标空间，适用于多目标优化问题。

**局限**：

1. **网格划分的复杂性**：随着目标数量的增加，网格数量呈指数级增长（`(nGrid + 2)^nObj`），导致存储和计算的复杂性显著增加，限制了PESA-II在高维目标空间中的应用。
2. **参数敏感性**：算法性能对参数设置（如`nGrid`、`InflationFactor`、`beta`等）较为敏感，需通过经验或试验调整以适应具体问题。
3. **计算开销**：网格创建和个体分配过程需要较多的计算资源，特别是在处理大规模种群和高维目标空间时，可能导致计算效率下降。

#### 五、应用场景

PESA-II适用于以下多目标优化问题：

1. **工程设计优化**：如结构设计、航空航天、机械设计等，需要同时优化多个性能指标。
2. **资源分配问题**：如项目管理、物流调度，需要在多个资源约束下进行优化。
3. **机器学习与数据挖掘**：如模型选择、特征选择，需要在准确性和复杂度之间寻求平衡。

#### 六、总结

PESA-II作为一种基于网格的进化多目标优化算法，通过引入网格划分和基于范围的选择策略，有效地维护了解集的多样性和覆盖性。然而，随着目标数量的增加，其计算复杂性和存储需求也显著增加。结合具体问题和应用需求，合理设置算法参数，并在必要时进行算法改进或结合其他优化策略，是提升PESA-II性能的关键。

我们可以清晰地看到PESA-II的各个组成部分是如何协同工作的，从初始化、支配关系判断、存档管理，到选择、交叉与变异操作，再到最终的结果展示。详细的中文注释有助于深入理解算法的实现细节，为进一步优化和应用PESA-II提供了坚实的基础。

---

### --- CreateGrid.m ---

```matlab
function [pop, grid]=CreateGrid(pop,nGrid,InflationFactor)
% CreateGrid 创建网格并将个体分配到相应的网格中
% 输入参数:
%   pop - 当前种群
%   nGrid - 每个目标维度的网格数量
%   InflationFactor - 网格膨胀因子
% 输出参数:
%   pop - 更新后的种群（包含网格索引）
%   grid - 网格结构数组

    % 提取种群中所有个体的目标值（成本）
    z = [pop.Cost]';
    
    % 计算每个目标的最小值和最大值
    zmin = min(z);
    zmax = max(z);
    
    % 计算每个目标的范围
    dz = zmax - zmin;
    
    % 计算膨胀后的最小值和最大值
    alpha = InflationFactor / 2;
    zmin = zmin - alpha * dz;
    zmax = zmax + alpha * dz;
    
    % 获取目标的数量
    nObj = numel(zmin);
    
    % 初始化每个目标的分割点
    C = zeros(nObj, nGrid + 3);
    for j = 1:nObj
        % 每个目标的分割点包括 -inf, 线性分割点, inf
        C(j, :) = [-inf, linspace(zmin(j), zmax(j), nGrid + 1), inf];
    end
    
    % 定义一个空的网格结构
    empty_grid.LB = [];        % 下界
    empty_grid.UB = [];        % 上界
    empty_grid.Index = [];     % 网格索引
    empty_grid.SubIndex = [];  % 网格的子索引（在每个目标维度上的位置）
    empty_grid.N = 0;          % 网格中的个体数量
    empty_grid.Members = [];   % 网格中的个体索引
    
    % 计算总网格数量
    nG = (nGrid + 2)^nObj;
    
    % 定义每个目标维度的网格大小
    GridSize = (nGrid + 2) * ones(1, nObj);
    
    % 初始化网格数组
    grid = repmat(empty_grid, nG, 1);
    
    % 为每个网格分配上下界
    for k = 1:nG
        SubIndex = cell(1, nObj);
        % 将线性索引转换为多维子索引
        [SubIndex{:}] = ind2sub(GridSize, k);
        SubIndex = cell2mat(SubIndex);
        
        % 记录网格的索引和子索引
        grid(k).Index = k;
        grid(k).SubIndex = SubIndex;
        
        % 初始化每个目标维度的下界和上界
        grid(k).LB = zeros(nObj, 1);
        grid(k).UB = zeros(nObj, 1);
        for j = 1:nObj
            grid(k).LB(j) = C(j, SubIndex(j));
            grid(k).UB(j) = C(j, SubIndex(j) + 1);
        end
    end
    
    % 将种群中的个体分配到相应的网格中
    [pop, grid] = FindPositionInGrid(pop, grid);
    
end
```

#### 注释说明

`CreateGrid.m` 文件的主要功能是创建多目标优化问题的网格结构，并将当前种群中的个体分配到相应的网格中。具体步骤包括：

1. **提取目标值**：从种群中提取所有个体的目标值（成本），并计算每个目标的最小值和最大值。
2. **网格范围膨胀**：根据膨胀因子`InflationFactor`，扩展每个目标的最小值和最大值，以确保边界外的个体也能被有效分配。
3. **网格划分**：为每个目标维度创建网格分割点，包括负无穷大和正无穷大作为边界。
4. **初始化网格结构**：定义一个空的网格结构，包含下界、上界、索引、子索引、个体数量和成员列表。
5. **网格分配**：通过循环为每个网格分配相应的上下界，并调用`FindPositionInGrid`函数将种群中的个体分配到对应的网格中。

该函数的输出是更新后的种群（包含每个个体的网格索引）和网格结构数组，后续的选择和截断操作将基于这些网格信息进行。

---

### --- Crossover.m ---

```matlab
function [y1, y2]=Crossover(x1, x2, params)
% Crossover 执行交叉操作，生成两个新的个体
% 输入参数:
%   x1, x2 - 父代个体的位置向量
%   params - 交叉操作的参数结构体，包含gamma, VarMin, VarMax
% 输出参数:
%   y1, y2 - 生成的子代个体的位置向量

    % 提取参数
    gamma = params.gamma;         % 控制交叉范围的参数
    VarMin = params.VarMin;       % 决策变量的下界
    VarMax = params.VarMax;       % 决策变量的上界
    
    % 生成交叉系数alpha，范围在[-gamma, 1+gamma]
    alpha = unifrnd(-gamma, 1 + gamma, size(x1));
    
    % 计算子代个体的位置
    y1 = alpha .* x1 + (1 - alpha) .* x2;
    y2 = alpha .* x2 + (1 - alpha) .* x1;
    
    % 确保子代的位置在允许范围内
    y1 = min(max(y1, VarMin), VarMax);
    y2 = min(max(y2, VarMin), VarMax);

end
```

#### 注释说明

`Crossover.m` 文件实现了交叉操作，用于生成新的子代个体。具体步骤包括：

1. **参数提取**：从`params`结构体中提取交叉参数，包括控制交叉范围的`gamma`以及决策变量的上下界`VarMin`和`VarMax`。
2. **生成交叉系数**：生成一个与父代个体位置向量相同尺寸的交叉系数`alpha`，其值在`[-gamma, 1 + gamma]`范围内随机分布。
3. **生成子代位置**：通过线性组合父代个体的位置向量`x1`和`x2`，生成两个子代个体的位置向量`y1`和`y2`。
4. **边界处理**：确保子代个体的位置向量在决策变量的允许范围内，通过`min`和`max`函数将其限制在`VarMin`和`VarMax`之间。

该函数的输出是两个新的子代个体的位置向量`y1`和`y2`，它们将在后续的变异操作或评价中被使用。

---

### --- DetermineDomination.m ---

```matlab
function pop=DetermineDomination(pop)
% DetermineDomination 确定种群中每个个体是否被支配
% 输入参数:
%   pop - 当前种群，包含目标值
% 输出参数:
%   pop - 更新后的种群，包含IsDominated标志

    n = numel(pop);  % 种群大小
    
    % 初始化每个个体的支配标志为false
    for i = 1:n
        pop(i).IsDominated = false;
    end
    
    % 双重循环，比较每对个体
    for i = 1:n
        if pop(i).IsDominated
            continue;  % 如果已被支配，跳过
        end
        
        for j = 1:n
            if Dominates(pop(j), pop(i))
                pop(i).IsDominated = true;  % 被j支配
                break;  % 不需要继续检查
            end
        end
    end

end
```

#### 注释说明

`DetermineDomination.m` 文件的功能是确定种群中每个个体是否被其他个体支配。具体步骤包括：

1. **初始化支配标志**：首先，将种群中所有个体的`IsDominated`标志初始化为`false`，表示默认没有被支配。
2. **支配关系判断**：通过双重循环，逐对比较种群中的个体。如果个体`j`支配个体`i`，则将`i`的`IsDominated`标志设置为`true`，并跳出内层循环，继续检查下一个个体。
3. **跳过已支配个体**：如果某个个体已经被标记为被支配，则跳过对其的进一步检查，提高计算效率。

该函数利用了辅助函数`Dominates.m`来判断两个个体之间的支配关系，最终输出更新后的种群，其中每个个体包含是否被支配的标志`IsDominated`。

---

### --- Dominates.m ---

```matlab
function b=Dominates(x, y)
% Dominates 判断个体x是否支配个体y
% 支配条件: x在所有目标上都不劣于y，且至少在一个目标上优于y
% 输入参数:
%   x, y - 两个待比较的个体，包含Cost字段
% 输出参数:
%   b - 布尔值，若x支配y，则为true，否则为false

    % 如果个体包含Cost字段，则提取目标值
    if isfield(x, 'Cost')
        x = x.Cost;
    end
    
    if isfield(y, 'Cost')
        y = y.Cost;
    end

    % 检查是否在所有目标上x <= y且至少一个目标上x < y
    b = all(x <= y) && any(x < y);

end
```

#### 注释说明

`Dominates.m` 文件实现了判断两个个体之间支配关系的功能。具体逻辑如下：

1. **提取目标值**：如果输入的个体`x`和`y`包含`Cost`字段，则提取其目标值向量进行比较。
2. **支配条件判断**：
   - **所有目标不劣**：检查个体`x`在所有目标上是否不劣于个体`y`，即`x`的每个目标值都小于或等于`y`对应的目标值。
   - **至少一个目标优于**：同时，要求在至少一个目标上，`x`的值严格小于`y`的值。
3. **返回结果**：如果以上两个条件同时满足，则返回`true`，表示`x`支配`y`；否则返回`false`。

该函数用于支配关系的判断，是非支配排序和支配关系筛选的基础。

---

### --- FindPositionInGrid.m ---

```matlab
function [pop, grid]=FindPositionInGrid(pop, grid)
% FindPositionInGrid 将种群中的个体分配到对应的网格中
% 输入参数:
%   pop - 当前种群，包含Cost字段
%   grid - 网格结构数组，包含LB和UB
% 输出参数:
%   pop - 更新后的种群，包含GridIndex
%   grid - 更新后的网格，包含成员信息

    % 提取所有网格的下界和上界
    LB = [grid.LB];
    UB = [grid.UB];
    
    % 初始化每个网格中的个体数量和成员列表
    for k = 1:numel(grid)
        grid(k).N = 0;
        grid(k).Members = [];
    end
    
    % 遍历种群中的每个个体，找到其所在的网格
    for i = 1:numel(pop)
        % 找到个体的网格索引
        k = FindGridIndex(pop(i).Cost, LB, UB);
        pop(i).GridIndex = k;  % 记录个体的网格索引
        
        % 更新网格中的个体数量和成员列表
        grid(k).N = grid(k).N + 1;
        grid(k).Members = [grid(k).Members, i];
    end

end

function k=FindGridIndex(z, LB, UB)
% FindGridIndex 根据个体的目标值z找到其所在的网格索引
% 输入参数:
%   z - 个体的目标值向量
%   LB, UB - 所有网格的下界和上界
% 输出参数:
%   k - 网格的线性索引

    nObj = numel(z);        % 目标数量
    
    nGrid = size(LB, 2);    % 每个目标的网格数量
    f = true(1, nGrid);     % 初始化筛选条件为全部真
    
    % 对每个目标，判断z是否在对应网格的范围内
    for j = 1:nObj
        f = f & (z(j) >= LB(j, :)) & (z(j) < UB(j, :));
    end
    
    % 找到满足所有目标条件的网格索引
    k = find(f);

end
```

#### 注释说明

`FindPositionInGrid.m` 文件的主要功能是将种群中的每个个体根据其目标值分配到相应的网格中，并更新网格的成员信息。具体步骤包括：

1. **提取网格边界**：从`grid`结构数组中提取所有网格的下界`LB`和上界`UB`。
2. **初始化网格信息**：遍历所有网格，初始化每个网格中的个体数量`N`为0，并清空成员列表`Members`。
3. **个体网格分配**：
   - 对于种群中的每个个体，调用`FindGridIndex`函数根据其目标值`z`找到对应的网格索引`k`。
   - 更新个体的`GridIndex`字段，记录其所在的网格索引。
   - 增加对应网格中的个体数量`N`，并将个体的索引添加到网格的`Members`列表中。
4. **辅助函数 `FindGridIndex`**：
   - 该函数根据个体的目标值`z`和所有网格的边界`LB`、`UB`，判断个体属于哪个网格。
   - 通过逐目标的比较，找到满足所有目标值在对应网格范围内的网格索引`k`。

最终，该文件输出更新后的种群（包含每个个体的网格索引）和网格结构数组（包含每个网格的成员信息），为后续的选择和截断操作提供基础。

---

### --- main.m ---

```matlab
% main.m 文件通常作为主入口调用其他函数
% 这里直接调用 pesa2.m 脚本
pesa2;
```

#### 注释说明

`main.m` 文件在此项目中充当主入口的角色，其主要功能是启动整个PESA-II算法的执行。具体实现非常简单，仅包含一行代码：

- **调用 `pesa2.m`**：通过直接调用`pesa2`脚本，启动多目标优化的主程序。

在实际应用中，`main.m` 文件可以进一步扩展，例如接受用户输入参数、配置不同的优化问题或参数设置等。但在当前实现中，它仅作为一个启动脚本，简化了运行过程。

---

### --- MOP2.m ---

```matlab
function z = MOP2(x)
% MOP2 定义了一个多目标优化问题（示例）
% 输入参数:
%   x - 决策变量向量
% 输出参数:
%   z - 目标值向量

    n = numel(x);  % 决策变量的数量
    
    % 计算第一个目标
    z1 = 1 - exp(-sum((x - 1 / sqrt(n)).^2));
    
    % 计算第二个目标
    z2 = 1 - exp(-sum((x + 1 / sqrt(n)).^2));
    
    % 返回目标值向量
    z = [z1
         z2];
    
end
```

#### 注释说明

`MOP2.m` 文件定义了一个具体的多目标优化问题（MOP），在此案例中为一个示例性的二目标问题。具体功能包括：

1. **输入参数**：接受一个决策变量向量`x`，其维度由问题定义（本例中默认为3维）。
2. **目标函数计算**：
   - **第一个目标 (`z1`)**：计算表达式`1 - exp(-sum((x - 1/sqrt(n)).^2))`，其目的是最小化个体与点`1/sqrt(n)`的距离。
   - **第二个目标 (`z2`)**：计算表达式`1 - exp(-sum((x + 1/sqrt(n)).^2))`，其目的是最小化个体与点`-1/sqrt(n)`的距离。
3. **输出目标值**：将两个目标值`z1`和`z2`组成向量`z`作为函数的输出。

该目标函数设计为两个相互竞争的目标，个体在优化过程中需要在这两个目标之间寻找平衡，实现Pareto最优解。

---

### --- Mutate.m ---

```matlab
function y = Mutate(x, params)
% Mutate 对个体x执行变异操作
% 输入参数:
%   x - 原始个体的位置向量
%   params - 变异操作的参数结构体，包含h, VarMin, VarMax
% 输出参数:
%   y - 变异后的个体的位置向量

    % 提取参数
    h = params.h;             % 变异步长因子
    VarMin = params.VarMin;   % 决策变量的下界
    VarMax = params.VarMax;   % 决策变量的上界
    
    % 计算变异的标准差
    sigma = h * (VarMax - VarMin);
    
    % 执行高斯变异
    y = x + sigma .* randn(size(x));
    
    % 或者执行均匀变异（注释掉）
    % y = x + sigma .* unifrnd(-1, 1, size(x));
    
    % 确保变异后的个体在允许范围内
    y = min(max(y, VarMin), VarMax);

end
```

#### 注释说明

`Mutate.m` 文件实现了变异操作，用于在进化过程中引入新的个体多样性。具体步骤包括：

1. **参数提取**：从`params`结构体中提取变异参数，包括变异步长因子`h`以及决策变量的上下界`VarMin`和`VarMax`。
2. **计算变异幅度**：根据`h`和决策变量的范围，计算变异的标准差`sigma`，用于控制变异的幅度。
3. **执行变异操作**：
   - **高斯变异**：通过向原始个体的位置向量`x`添加服从正态分布的随机噪声，实现变异。
   - **均匀变异（可选）**：注释掉的代码提供了另一种变异方式，通过添加均匀分布的随机噪声实现变异。
4. **边界处理**：确保变异后的个体位置向量`y`仍然在决策变量的允许范围内，通过`min`和`max`函数将其限制在`VarMin`和`VarMax`之间。

该函数的输出是变异后的个体位置向量`y`，它将在后续的评价和选择过程中被使用。

---

### --- pesa2.m ---

```matlab
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
```

#### 注释说明

`pesa2.m` 文件是PESA-II算法的主脚本，负责整个多目标优化过程的执行。主要功能包括初始化、主循环（包含支配关系判断、存档管理、网格创建、交叉和变异操作）以及结果展示。具体步骤如下：

1. **初始化**：
   - **清理环境**：通过`clc`, `clear`, `close all`清理命令窗口、工作区变量和图形窗口。
   - **问题定义**：设置目标函数`MOP2`，定义决策变量的数量、范围以及目标的数量。
   - **PESA-II 参数设置**：设置最大迭代次数、种群大小、存档大小、网格数量、膨胀因子、选择和删除参数、交叉和变异概率及其相关参数。
   - **种群初始化**：生成初始种群，随机初始化个体的位置，并计算其目标值。存档初始化为空。

2. **主循环**（迭代执行`MaxIt`次）：
   - **支配关系判断**：通过`DetermineDomination`函数确定种群中每个个体是否被支配。
   - **非支配个体提取**：提取非被支配的个体并添加到存档中。
   - **存档支配关系判断**：更新存档中每个个体的支配关系，提取非被支配的存档个体作为当前Pareto前沿。
   - **网格创建与分配**：调用`CreateGrid`函数创建网格并将存档中的个体分配到对应的网格中。
   - **存档截断**：如果存档大小超过设定的`nArchive`，通过`TruncatePopulation`函数删除多余的个体，并重新创建网格。
   - **Pareto前沿绘制**：使用`PlotCosts`函数绘制当前的Pareto前沿。
   - **迭代信息显示**：在命令窗口显示当前迭代次数及Pareto前沿的成员数量。
   - **交叉与变异操作**：
     - **交叉**：根据交叉概率和次数，选择父代个体进行交叉，生成子代个体，并计算其目标值。
     - **变异**：根据变异概率和次数，选择父代个体进行变异，生成子代个体，并计算其目标值。
   - **更新种群**：将交叉和变异生成的子代添加到种群中，为下一次迭代做准备。

3. **结果展示**：
   - **统计信息**：计算并显示Pareto前沿各目标的最小值、最大值、范围、标准差和均值。
   - **绘图结果**：通过图形窗口展示最终的Pareto前沿分布。

该脚本通过综合运用支配关系判断、网格管理和遗传操作，实现了PESA-II算法在多目标优化问题上的求解过程。

---

### --- PlotCosts.m ---

```matlab
function PlotCosts(PF)
% PlotCosts 绘制Pareto前沿的目标值
% 输入参数:
%   PF - Pareto前沿的个体数组，包含Cost字段

    % 提取所有个体的目标值
    PFC = [PF.Cost];
    
    % 绘制第一个目标与第二个目标的关系
    plot(PFC(1, :), PFC(2, :), 'x');
    xlabel('1^{st} Objective');  % x轴标签
    ylabel('2^{nd} Objective');  % y轴标签
    grid on;                     % 显示网格

end
```

#### 注释说明

`PlotCosts.m` 文件的功能是绘制Pareto前沿的目标值，帮助可视化多目标优化的结果。具体步骤包括：

1. **提取目标值**：从Pareto前沿个体数组`PF`中提取所有个体的目标值，组成目标值矩阵`PFC`。
2. **绘制图形**：
   - 使用`plot`函数将第一个目标与第二个目标的值绘制为散点图，标记样式为`'x'`。
   - 设置x轴和y轴的标签，分别为“第一个目标”和“第二个目标”。
   - 启用网格显示，便于观察数据分布。

该函数适用于二目标优化问题，通过可视化Pareto前沿，直观展示优化结果的分布和多样性。如果优化问题具有更多目标，可以扩展该函数以支持多维度的可视化或采用其他适合的可视化技术。

---

### --- RouletteWheelSelection.m ---

```matlab
function i = RouletteWheelSelection(p)
% RouletteWheelSelection 使用轮盘赌选择法选择个体
% 输入参数:
%   p - 每个个体的选择概率向量
% 输出参数:
%   i - 被选择的个体的索引

    r = rand * sum(p);         % 生成一个0到总概率之间的随机数
    c = cumsum(p);             % 计算累积概率
    i = find(r <= c, 1, 'first');  % 找到第一个累积概率大于或等于r的位置

end
```

#### 注释说明

`RouletteWheelSelection.m` 文件实现了轮盘赌选择法，用于根据概率分布选择个体。具体步骤包括：

1. **随机数生成**：生成一个在`[0, sum(p)]`范围内的随机数`r`，其中`p`是各个个体的选择概率向量。
2. **累积概率计算**：计算累积概率向量`c`，即`p`的累积和。
3. **选择个体**：通过`find`函数找到第一个满足`r <= c`的索引`i`，即被选择的个体。

轮盘赌选择法根据个体的概率分布随机选择个体，概率越高的个体被选中的可能性越大。该方法广泛应用于遗传算法和进化算法中的选择操作，确保了优良个体的繁殖机会，同时保持种群的多样性。

---

### --- SelectFromPopulation.m ---

```matlab
function P = SelectFromPopulation(pop, grid, beta)
% SelectFromPopulation 从种群中选择一个个体，基于网格和选择参数beta
% 输入参数:
%   pop - 当前种群
%   grid - 网格结构数组
%   beta - 选择操作的参数
% 输出参数:
%   P - 被选择的个体

    % 筛选出非空网格
    sg = grid([grid.N] > 0);
    
    % 计算每个网格的选择概率，基于网格中个体数量的倒数的beta次方
    p = 1 ./ [sg.N].^beta;
    p = p / sum(p);  % 归一化概率
    
    % 使用轮盘赌选择一个网格
    k = RouletteWheelSelection(p);
    
    % 获取被选网格中的成员索引
    Members = sg(k).Members;
    
    % 从成员中随机选择一个个体
    i = Members(randi([1, numel(Members)]));
    
    % 返回被选择的个体
    P = pop(i);

end
```

#### 注释说明

`SelectFromPopulation.m` 文件实现了从种群中选择个体的功能，选择过程基于网格信息和参数`beta`。具体步骤包括：

1. **筛选非空网格**：从网格结构数组`grid`中筛选出包含至少一个个体的网格`sg`。
2. **计算选择概率**：
   - 计算每个非空网格的选择概率`p`，其值与网格中个体数量的倒数的`beta`次方成正比，即网格中个体越少，其选择概率越高。
   - 对概率向量`p`进行归一化，使其和为1。
3. **网格选择**：使用轮盘赌选择法`RouletteWheelSelection`从非空网格中选择一个网格索引`k`。
4. **个体选择**：
   - 获取被选网格`sg(k)`中的成员索引`Members`。
   - 从成员列表中随机选择一个个体索引`i`，并返回该个体`P`。

该函数的设计旨在优先选择稀疏网格中的个体，从而促进种群的多样性和覆盖性。参数`beta`控制选择概率与网格稀疏程度之间的关系，`beta`越大，稀疏网格的优先级越高。

---

### --- TruncatePopulation.m ---

```matlab
function [pop, grid] = TruncatePopulation(pop, grid, E, beta)
% TruncatePopulation 截断种群以满足存档大小限制
% 输入参数:
%   pop - 当前存档
%   grid - 网格结构数组
%   E - 需要删除的个体数量
%   beta - 删除操作的参数
% 输出参数:
%   pop - 更新后的存档
%   grid - 更新后的网格

    ToBeDeleted = [];  % 初始化待删除个体的索引列表
    
    for e = 1:E
        % 筛选出非空网格
        sg = grid([grid.N] > 0);
        
        % 计算每个网格的选择概率，基于网格中个体数量的beta次方
        p = [sg.N].^beta;
        p = p / sum(p);  % 归一化概率
        
        % 使用轮盘赌选择一个网格
        k = RouletteWheelSelection(p);
        
        % 获取被选网格中的成员索引
        Members = sg(k).Members;
        
        % 从成员中随机选择一个个体进行删除
        i = Members(randi([1, numel(Members)]));
        
        % 移除被删除的个体索引
        Members(Members == i) = [];
        
        % 更新网格中的成员列表和个体数量
        grid(sg(k).Index).Members = Members;
        grid(sg(k).Index).N = numel(Members);
        
        % 记录待删除的个体索引
        ToBeDeleted = [ToBeDeleted, i]; %#ok
        
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA123
% Project Title: Pareto Envelope-based Selection Algorithm II (PESA-II)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%
    end
    
    % 从存档中删除选定的个体
    pop(ToBeDeleted) = [];
    
    % 注释信息（保留版权信息）

end
```

#### 注释说明

`TruncatePopulation.m` 文件的主要功能是在存档大小超过预定限制时，截断多余的个体以满足存档容量。具体步骤包括：

1. **初始化待删除列表**：创建一个空数组`ToBeDeleted`，用于记录需要删除的个体索引。
2. **循环删除操作**（执行`E`次）：
   - **筛选非空网格**：从网格结构数组`grid`中筛选出包含至少一个个体的网格`sg`。
   - **计算删除概率**：
     - 计算每个非空网格的选择概率`p`，其值与网格中个体数量的`beta`次方成正比，即网格中个体越多，其被选择删除的概率越高。
     - 对概率向量`p`进行归一化，使其和为1。
   - **网格选择**：使用轮盘赌选择法`RouletteWheelSelection`从非空网格中选择一个网格索引`k`。
   - **个体选择与删除**：
     - 获取被选网格`sg(k)`中的成员索引`Members`。
     - 从成员列表中随机选择一个个体索引`i`，并将其添加到待删除列表`ToBeDeleted`中。
     - 从网格成员列表中移除被删除的个体索引，并更新网格中的个体数量`N`。
3. **删除个体**：通过索引`ToBeDeleted`从存档`pop`中删除选定的个体。
4. **版权信息**：保留了原始代码中的版权声明，确保版权信息得到尊重。

该函数通过基于网格的删除策略，优先删除拥挤程度较高的网格中的个体，从而保持存档的多样性和覆盖性。参数`beta`控制删除概率与网格个体数量之间的关系，`beta`越大，个体数量多的网格被优先删除的可能性越高。

---

### --- ZDT.m ---

```matlab
function z = ZDT(x)
% ZDT 定义了ZDT多目标优化问题（具体版本未明确）
% 输入参数:
%   x - 决策变量向量
% 输出参数:
%   z - 目标值向量

    n = numel(x);  % 决策变量的数量
    
    % 第一个目标函数
    f1 = x(1);
    
    % 计算辅助函数g
    g = 1 + 9 / (n - 1) * sum(x(2:end));
    
    % 计算辅助函数h
    h = 1 - sqrt(f1 / g);
    
    % 第二个目标函数
    f2 = g * h;
    
    % 返回目标值向量
    z = [f1
         f2];
    
end
```

#### 注释说明

`ZDT.m` 文件定义了ZDT系列中的一个多目标优化问题，具体版本未在代码中明确（ZDT1、ZDT2等）。ZDT问题是经典的多目标优化测试问题，常用于算法性能评估。具体步骤如下：

1. **输入参数**：接受一个决策变量向量`x`，其维度由问题定义（常见为30维）。
2. **目标函数计算**：
   - **第一个目标函数 (`f1`)**：直接取决策变量向量`x`的第一个元素，即`f1 = x(1)`。
   - **辅助函数 `g`**：计算公式为`g = 1 + 9/(n-1) * sum(x(2:end))`，表示除第一个决策变量外，其余变量的和经过线性变换。
   - **辅助函数 `h`**：计算公式为`h = 1 - sqrt(f1/g)`，表示与第一个目标函数和辅助函数`g`相关的关系。
   - **第二个目标函数 (`f2`)**：由辅助函数`g`和`h`计算得出，即`f2 = g * h`。
3. **输出目标值**：将两个目标值`f1`和`f2`组成向量`z`作为函数的输出。

ZDT问题的设计旨在生成具有不同形状和难度的Pareto前沿，适用于测试和比较多目标优化算法的性能。根据不同版本，ZDT问题可能具有不同的`h`函数形式，从而影响Pareto前沿的形状（如凸、凹、非凸等）。

---
