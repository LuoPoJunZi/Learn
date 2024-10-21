# MOGWO算法（多目标灰狼优化算法）

MOGWO算法（Multi-Objective Grey Wolf Optimizer，多目标灰狼优化算法）是一种基于自然界狼群捕猎行为的多目标优化算法。MOGWO是GWO（Grey Wolf Optimizer，灰狼优化算法）的扩展，用于解决具有多个目标的优化问题。

### MOGWO算法的工作原理

1. **灰狼的社会层级**：
   - 狼群的社会层级分为4类：
     - α狼（领导者）：负责做出决策。
     - β狼（次级领导者）：辅助α狼，协助决策。
     - δ狼（普通成员）：受α和β的支配，并指导最低级别的狼。
     - ω狼（最低层次狼）：遵循其他狼的命令。
   
2. **捕猎过程的模拟**：
   - MOGWO模拟了灰狼群捕猎的三个主要阶段：
     1. **围猎**：狼群围绕猎物移动，并不断逼近目标。
     2. **追捕**：α、β、δ狼负责追捕猎物，基于狼的位置不断更新解。
     3. **攻击**：当狼群接近猎物时，狼群通过收敛因子来调整捕猎行为，最终收敛到最优解。

3. **多目标处理**：
   - 在MOGWO中，多个目标函数同时进行优化。通常，通过帕累托前沿（Pareto Front）来选择多目标下的最优解。
   - MOGWO会同时考虑多个目标，并通过进化过程来逼近Pareto最优解集，得到不同目标之间的权衡。

4. **拥挤距离计算**：
   - 为了保持解的多样性，MOGWO通常采用拥挤距离（Crowding Distance）来评价解的分布，确保解集在目标空间中的分布均匀。
   
5. **帕累托支配排序**：
   - 在每一代中，MOGWO根据帕累托支配关系对解进行排序，并采用非支配排序法来生成Pareto前沿，从而得到一组不被其他解支配的最优解。

### MOGWO的MATLAB实现
MOGWO算法的实现通常包括以下步骤：

1. **初始化狼群**：
   - 随机生成初始解集，初始化多个狼的位置（解的表示），并计算目标函数值。

2. **非支配排序**：
   - 对初始解集进行非支配排序，识别出当前的Pareto前沿。

3. **捕猎行为模拟**：
   - 持续更新狼的位置，通过模拟狼群的围捕、追捕和攻击行为，使解不断逼近最优。

4. **更新帕累托前沿**：
   - 每一代更新Pareto前沿，保存非支配解。

5. **终止条件**：
   - 当满足最大迭代次数或解集收敛到一定程度时，算法终止，输出Pareto前沿的解集。

MOGWO算法通常用于多目标优化问题，例如多目标的工程设计、经济调度等。其MATLAB实现可以在多个开源库中找到，也可以根据灰狼优化算法的基础代码进行扩展。


## `CreateEmptyParticle.m` 文件的详细中文注释版本：

```matlab

% 函数 CreateEmptyParticle
% 用于创建一个或多个空的粒子结构体

function particle=CreateEmptyParticle(n)
    
    % 如果没有传入参数 n，默认创建一个粒子
    if nargin<1
        n=1;
    end

    % 创建一个空粒子结构体
    empty_particle.Position=[];      % 粒子的位置向量
    empty_particle.Velocity=[];      % 粒子的速度向量
    empty_particle.Cost=[];          % 粒子的目标函数值（成本）
    empty_particle.Dominated=false;  % 是否被其他粒子支配的标志
    empty_particle.Best.Position=[]; % 粒子迄今为止的最佳位置
    empty_particle.Best.Cost=[];     % 粒子迄今为止的最佳目标函数值
    empty_particle.GridIndex=[];     % 粒子所在网格的索引
    empty_particle.GridSubIndex=[];  % 粒子所在网格的子索引
    
    % 使用 repmat 函数复制 n 个粒子结构体
    particle=repmat(empty_particle,n,1);
    
end
```

### 注释说明：
1. **粒子属性解释**：
   - `Position`：粒子在解空间中的当前位置。
   - `Velocity`：粒子当前的速度。
   - `Cost`：粒子当前的目标函数值，即它在优化问题中的“成本”。
   - `Dominated`：布尔变量，表示粒子是否被其他粒子支配（支配指的是另一个粒子在所有目标上都不比当前粒子差）。
   - `Best.Position`：粒子到目前为止找到的最优解的位置。
   - `Best.Cost`：粒子到目前为止找到的最优解的目标函数值。
   - `GridIndex` 和 `GridSubIndex`：这些用于多目标粒子在网格中的位置标识，通常用于非支配排序和拥挤距离计算。

2. **代码逻辑**：
   - 如果未传递 `n` 参数，默认创建一个空粒子。
   - 使用 `repmat` 函数创建多个粒子副本（当 `n>1` 时），便于初始化整个粒子群。


## `CreateHypercubes.m` 文件的详细中文注释版本：

```matlab

% 函数 CreateHypercubes
% 用于为每个目标生成超立方体（或网格），这些网格用于非支配解的分类。

function G=CreateHypercubes(costs, ngrid, alpha)

    % 获取目标数目，即成本矩阵的行数
    nobj = size(costs, 1);
    
    % 创建一个空的网格结构体，用于存储每个目标的上下界
    empty_grid.Lower = [];    % 网格的下边界
    empty_grid.Upper = [];    % 网格的上边界
    G = repmat(empty_grid, nobj, 1);  % 为每个目标分配一个网格

    % 遍历每个目标
    for j = 1:nobj
        
        % 获取第 j 个目标的最小值和最大值
        min_cj = min(costs(j, :));
        max_cj = max(costs(j, :));
        
        % 计算扩展范围 dcj（扩展比例为 alpha）
        dcj = alpha * (max_cj - min_cj);
        
        % 扩展目标值的上下界
        min_cj = min_cj - dcj;
        max_cj = max_cj + dcj;
        
        % 使用 linspace 函数将目标值范围划分为 ngrid-1 个网格
        gx = linspace(min_cj, max_cj, ngrid - 1);
        
        % 设置第 j 个目标的网格上下界
        G(j).Lower = [-inf gx];   % 下边界数组，第一项为 -inf
        G(j).Upper = [gx inf];    % 上边界数组，最后一项为 inf
        
    end

end
```

### 注释说明：
1. **`costs`**: 这是目标值矩阵，其中每一行表示一个目标，列表示不同解的目标值。
2. **`ngrid`**: 指定每个目标划分的网格数目。
3. **`alpha`**: 这是一个比例因子，用于扩展每个目标值的范围，确保边界外有一定的缓冲区域。
4. **超立方体生成**:
   - 函数为每个目标生成一个包含 `ngrid` 个区间的网格。每个网格有上下边界，分别存储在 `G(j).Lower` 和 `G(j).Upper` 中。
   - `linspace(min_cj, max_cj, ngrid-1)` 用于均匀地在扩展后的目标范围内生成 `ngrid-1` 个分隔点，从而创建多个网格。

5. **无穷大边界**:
   - 为了确保每个网格都能覆盖完整的目标值域，最左边的下界设为 `-inf`，最右边的上界设为 `inf`，防止目标值落在边界外。

## `DeleteFromRep.m` 文件的详细中文注释版本：

```matlab

% 函数 DeleteFromRep
% 该函数用于从外部存档 (rep) 中删除额外的解（粒子）。
% 参数：
%   - rep: 外部存档，即当前存储的非支配解的集合
%   - EXTRA: 需要删除的解的数量
%   - gamma: 控制删除策略的参数，默认为 1

function rep = DeleteFromRep(rep, EXTRA, gamma)

    % 如果没有传入 gamma 参数，则默认值为 1
    if nargin < 3
        gamma = 1;
    end

    % 循环执行 EXTRA 次，逐步从存档中删除解
    for k = 1:EXTRA
        % 获取已占据网格的索引及其成员数量
        [occ_cell_index, occ_cell_member_count] = GetOccupiedCells(rep);

        % 计算删除概率 p，成员数越多，删除的概率越大（由 gamma 控制）
        p = occ_cell_member_count .^ gamma;  % 提升成员数的次幂
        p = p / sum(p);  % 归一化为概率分布

        % 使用轮盘赌法选择一个网格进行删除操作
        selected_cell_index = occ_cell_index(RouletteWheelSelection(p));

        % 提取所有存档粒子的网格索引
        GridIndices = [rep.GridIndex];

        % 找到当前被选中的网格中所有的粒子
        selected_cell_members = find(GridIndices == selected_cell_index);

        % 获取该网格中的粒子数量
        n = numel(selected_cell_members);

        % 随机选择该网格中的一个粒子进行删除
        selected_memebr_index = randi([1 n]);

        % 获取该粒子在存档中的实际索引
        j = selected_cell_members(selected_memebr_index);
        
        % 删除该粒子，从存档中移除
        rep = [rep(1:j-1); rep(j+1:end)];
    end
    
end
```

### 注释说明：
1. **`DeleteFromRep`**:
   - 该函数用于从存档（`rep`）中删除多余的解，以保持存档大小不超过预定义的上限。
   - 当存档中的解（非支配解）数量超过允许的上限时，需要通过特定策略删除一些解，确保存档大小合理。

2. **参数说明**：
   - `EXTRA`: 需要删除的粒子数量，即存档中超出部分的解。
   - `gamma`: 影响删除概率的参数。`gamma` 控制成员数量与删除概率的关系，值越大，成员较多的网格越可能被选择。

3. **主要步骤**：
   - **获取已占据网格**：首先通过 `GetOccupiedCells` 函数获取哪些网格被粒子占据，以及每个网格中的粒子数量。
   - **计算删除概率**：使用成员数量的 `gamma` 次幂来计算每个网格被选中的概率。网格中粒子越多，被选择的概率就越高。
   - **选择粒子删除**：通过轮盘赌选择一个网格，随后随机选择该网格中的一个粒子进行删除。

4. **删除逻辑**：
   - 该函数通过 `randi` 随机选择一个粒子，并通过调整 `rep` 数组来移除该粒子。

## `DetermineDomination.m` 文件的详细中文注释版本：

```matlab
% 函数 DetermineDomination
% 该函数用于确定种群中每个解的支配关系，即判断每个个体是否被其他个体支配。
% 这一步对于多目标优化非常重要，因为它决定了哪些解是非支配的，哪些解将被淘汰。
% 参数：
%   - pop: 种群，包含多个个体，每个个体有自己的目标值和支配状态
% 返回：
%   - pop: 更新后的种群，其中每个个体的支配状态（Dominated）已确定

function pop = DetermineDomination(pop)

    % 获取种群中个体的数量
    npop = numel(pop);
    
    % 遍历种群中的每个个体 i
    for i = 1:npop
        % 首先假设个体 i 未被支配
        pop(i).Dominated = false;
        
        % 与种群中的其他个体 j 进行比较（只比较 i 之前的个体）
        for j = 1:i-1
            % 只有当个体 j 未被支配时，才需要进行支配关系的检查
            if ~pop(j).Dominated
                % 如果个体 i 支配个体 j，则标记 j 被支配
                if Dominates(pop(i), pop(j))
                    pop(j).Dominated = true;
                % 如果个体 j 支配个体 i，则标记 i 被支配，并停止进一步比较
                elseif Dominates(pop(j), pop(i))
                    pop(i).Dominated = true;
                    break;  % 一旦 i 被支配，不再需要与其他个体比较
                end
            end
        end
    end

end
```

### 注释说明：
1. **`DetermineDomination`**：
   - 该函数用于检查多目标优化中种群（`pop`）的支配关系。所谓支配，通常指一个解在所有目标上都优于或等于另一个解，并至少在一个目标上优于另一个解。
   
2. **支配关系**：
   - **`Dominates(pop(i), pop(j))`**：这个函数检查个体 `i` 是否支配个体 `j`。如果 `i` 支配 `j`，那么 `j` 会被标记为已被支配（`Dominated = true`）。
   - 支配关系的确定是基于目标值的比较，支配的个体将被淘汰，不会作为下一代的一部分。

3. **算法逻辑**：
   - 对于每个个体 `i`，与它之前的所有个体 `j` 进行比较（`for j=1:i-1`）。这个方法通过减少比较次数来提高效率。
   - 当个体 `i` 支配 `j` 时，将 `j` 标记为被支配。如果 `j` 支配 `i`，则 `i` 被标记为支配，直接结束对 `i` 的比较，跳到下一个个体。

4. **优化机制**：
   - 一旦某个个体被另一个个体支配，便不再与其他个体进行比较，这种机制提高了计算效率。

## `Dominates.m` 文件的详细中文注释版本：

```matlab
% 函数 Dominates
% 该函数用于判断解 x 是否支配解 y。支配是多目标优化中的核心概念，
% 一个解 x 支配解 y 当且仅当：
% 1. 解 x 在所有目标上不劣于解 y；
% 2. 解 x 在至少一个目标上优于解 y。
% 参数：
%   - x: 解 x，包含目标值 (Cost)，可以是结构体也可以是向量
%   - y: 解 y，包含目标值 (Cost)，可以是结构体也可以是向量
% 返回：
%   - dom: 布尔值，若 x 支配 y，返回 true；否则返回 false

function dom = Dominates(x, y)

    % 如果 x 是结构体，则提取其目标值 (Cost)
    if isstruct(x)
        x = x.Cost;
    end

    % 如果 y 是结构体，则提取其目标值 (Cost)
    if isstruct(y)
        y = y.Cost;
    end
    
    % 判断支配条件：
    % 1. all(x <= y): 确保 x 在所有目标上不劣于 y
    % 2. any(x < y): 确保 x 在至少一个目标上优于 y
    dom = all(x <= y) && any(x < y);

end
```

### 注释说明：
1. **`Dominates`**：
   - 该函数判断解 `x` 是否支配解 `y`，这是多目标优化中的基本比较规则之一。
   - **支配规则**：
     1. `x` 在所有目标上不劣于 `y`（即目标值相等或更优）；
     2. `x` 在至少一个目标上优于 `y`。

2. **参数处理**：
   - `x` 和 `y` 可以是结构体（通常包含多个字段，例如 `Cost`），也可以是直接的向量。
   - 如果是结构体，函数通过提取 `Cost` 字段来进行比较；如果是向量，则直接使用目标值进行比较。

3. **逻辑条件**：
   - **`all(x <= y)`**：这确保 `x` 的目标值在所有维度上都小于或等于 `y` 的目标值。
   - **`any(x < y)`**：这确保 `x` 的目标值在至少一个维度上小于 `y` 的目标值。


## `GetCosts.m` 文件的详细中文注释版本：

```matlab
% 函数 GetCosts
% 该函数用于从种群中提取所有个体的目标值 (Cost)，并将其组织为一个矩阵。
% 参数：
%   - pop: 种群，包含多个个体，每个个体具有一个目标值 (Cost) 向量
% 返回：
%   - costs: 一个矩阵，其中每一列对应一个个体的目标值，每一行对应一个目标

function costs = GetCosts(pop)

    % 获取每个个体的目标数量，即目标值 (Cost) 向量的维度
    nobj = numel(pop(1).Cost);

    % 将种群中的所有目标值提取并重组为一个 nobj 行的矩阵
    % 每列表示一个个体的目标值，每行表示某个目标在所有个体中的值
    costs = reshape([pop.Cost], nobj, []);

end
```

### 注释说明：
1. **`GetCosts`**：
   - 该函数从种群 `pop` 中提取所有个体的目标值，形成一个矩阵以便进一步操作。例如，在多目标优化中，我们通常需要对所有个体的目标值进行处理和分析。

2. **函数逻辑**：
   - 首先，通过 `numel(pop(1).Cost)` 获取每个个体的目标数量（`nobj`），假设所有个体的目标数量是相同的。
   - 接着，利用 MATLAB 的 `reshape` 函数，将所有个体的目标值组织为一个 `nobj × npop` 的矩阵，其中 `npop` 是种群中的个体数。

3. **矩阵形式**：
   - 输出的 `costs` 是一个矩阵，每列表示一个个体的目标值，每行表示某个目标在所有个体中的值。例如，`costs(1,:)` 是所有个体在第一个目标上的值。

此函数非常简洁地实现了提取目标值的功能，有助于后续的优化和分析步骤。

## `GetGridIndex.m` 文件的详细中文注释版本：

```matlab
% 函数 GetGridIndex
% 该函数用于获取粒子在超立方体网格中的索引。网格索引用于确定粒子属于哪个网格单元，
% 以便在多目标优化算法中进行存储和选择操作。
% 参数：
%   - particle: 包含目标值 (Cost) 的粒子
%   - G: 网格结构体数组，其中每个元素包含当前维度的网格边界
% 返回：
%   - Index: 粒子的全局网格索引，用于标识粒子在哪个网格单元中
%   - SubIndex: 每个目标的子网格索引，用于标识粒子在每个目标维度的网格位置

function [Index, SubIndex] = GetGridIndex(particle, G)

    % 获取粒子的目标值向量 (Cost)
    c = particle.Cost;
    
    % 获取目标的数量 (nobj) 和网格单元的数量 (ngrid)
    nobj = numel(c);
    ngrid = numel(G(1).Upper);
    
    % 初始化生成网格索引的表达式，使用 MATLAB 函数 sub2ind
    str = ['sub2ind(' mat2str(ones(1,nobj) * ngrid)];
    
    % 初始化子索引数组 SubIndex，用于存储每个目标的网格索引
    SubIndex = zeros(1, nobj);
    
    % 对每个目标维度进行处理
    for j = 1:nobj
        
        % 获取第 j 个目标的网格边界 (Upper)
        U = G(j).Upper;
        
        % 找到目标值 c(j) 所对应的网格单元的索引
        i = find(c(j) < U, 1, 'first');
        
        % 将该维度的网格索引存储到 SubIndex 数组中
        SubIndex(j) = i;
        
        % 将该维度的索引 i 添加到 str 表达式中，用于计算全局索引
        str = [str ',' num2str(i)];
    end
    
    % 完成用于计算全局索引的表达式
    str = [str ');'];
    
    % 通过 eval 函数执行生成的字符串，计算粒子的全局网格索引
    Index = eval(str);
    
end
```

### 注释说明：
1. **`GetGridIndex`**：
   - 该函数的作用是根据粒子的目标值确定其在网格中的位置，包括每个目标维度的子网格索引（`SubIndex`）和粒子在整个网格系统中的全局索引（`Index`）。
   - 这在基于网格的多目标优化算法中非常重要，网格索引用于引导优化过程中的存储、选择和比较。

2. **函数逻辑**：
   - `nobj` 是目标的数量，`ngrid` 是每个目标维度的网格数量。
   - **`SubIndex`**：为每个目标找到其对应的网格单元的索引。
   - **`Index`**：通过 `sub2ind` 函数计算粒子在整个多维网格系统中的全局索引，用于标识该粒子属于哪个超立方体。

3. **`sub2ind`**：
   - `sub2ind` 是 MATLAB 的一个函数，用于将多维数组的子索引转换为线性索引。在这里，`str` 是生成 `sub2ind` 的表达式，最终通过 `eval` 执行，得到粒子的全局索引 `Index`。

此函数帮助在网格系统中高效地定位粒子的位置，便于算法进行选择和优化操作。

## `GetNonDominatedParticles.m` 文件的详细中文注释版本：

```matlab
% 函数 GetNonDominatedParticles
% 该函数用于从种群中提取非支配粒子，这些粒子在多目标优化中是最优解集的一部分。
% 参数：
%   - pop: 种群，包含多个个体，每个个体都有一个布尔属性 Dominated
% 返回：
%   - nd_pop: 非支配粒子的集合，即不被其他粒子支配的粒子

function nd_pop = GetNonDominatedParticles(pop)

    % 使用逻辑索引提取未被支配的粒子
    % ND 是一个布尔数组，表示哪些粒子未被其他粒子支配
    ND = ~[pop.Dominated];
    
    % 根据 ND 的布尔值提取非支配粒子
    nd_pop = pop(ND);

end
```

### 注释说明：
1. **`GetNonDominatedParticles`**：
   - 该函数的作用是从给定的种群中提取出非支配的粒子，非支配粒子通常表示了当前的帕累托前沿（Pareto Front），在多目标优化中非常重要。

2. **逻辑处理**：
   - **`[pop.Dominated]`**：这行代码将种群中每个个体的 `Dominated` 属性提取为一个布尔数组。若个体被支配则对应值为 `true`，否则为 `false`。
   - **`ND = ~[pop.Dominated]`**：通过逻辑取反 (`~`)，得到非支配粒子的索引。

3. **结果提取**：
   - 最后，通过逻辑索引 `ND` 从原种群 `pop` 中提取非支配粒子，形成新的种群 `nd_pop`，这就是最终返回的结果。

此函数在多目标优化中用于收集当前非支配解，有助于后续的优化和选择操作。

## `GetOccupiedCells.m` 文件的详细中文注释版本：

```matlab
% 函数 GetOccupiedCells
% 该函数用于获取在种群中被占用的网格单元及其成员数量。这对于了解哪些网格单元被粒子占据是重要的。
% 参数：
%   - pop: 种群，包含多个个体，每个个体都有一个网格索引 (GridIndex)
% 返回：
%   - occ_cell_index: 被占用的网格单元索引的数组
%   - occ_cell_member_count: 每个被占用网格单元中粒子的数量

function [occ_cell_index, occ_cell_member_count] = GetOccupiedCells(pop)

    % 提取所有个体的网格索引，形成一个一维数组
    GridIndices = [pop.GridIndex];
    
    % 获取所有独特的网格索引，表示被占用的网格单元
    occ_cell_index = unique(GridIndices);
    
    % 初始化被占用网格单元中成员数量的数组
    occ_cell_member_count = zeros(size(occ_cell_index));

    % 获取被占用网格单元的数量
    m = numel(occ_cell_index);
    
    % 对每个被占用的网格单元计数其成员数量
    for k = 1:m
        % 计算当前网格单元中粒子的数量
        occ_cell_member_count(k) = sum(GridIndices == occ_cell_index(k));
    end
    
end
```

### 注释说明：
1. **`GetOccupiedCells`**：
   - 该函数的目的是识别出种群中被占用的网格单元及其各自的成员数量。了解网格的占用情况对于多目标优化中的选择和更新策略至关重要。

2. **逻辑处理**：
   - **`GridIndices = [pop.GridIndex]`**：将种群中每个粒子的网格索引提取到一个数组 `GridIndices` 中。
   - **`occ_cell_index = unique(GridIndices)`**：使用 `unique` 函数找出所有唯一的网格索引，表示当前被占用的网格单元。

3. **成员计数**：
   - `occ_cell_member_count` 用于存储每个占用网格单元中粒子的数量。通过循环，计算每个独特网格单元中粒子的数量，最终形成 `occ_cell_member_count` 数组。

此函数在处理网格相关的多目标优化算法时，能够帮助算法了解每个网格单元的占用状态，有助于算法的选择和更新。

## `MOGWo.m` 文件的详细中文注释版本：

```matlab
% 清除工作区变量和命令窗口
clear all
clc

% 设置绘图标志，1表示绘图
drawing_flag = 1;

% 定义变量数量
nVar = 5;

% 目标函数，使用 ZDT3 函数进行优化
fobj = @(x) ZDT3(x);

% 定义下界和上界
lb = zeros(1, 5);  % 下界为0
ub = ones(1, 5);   % 上界为1

% 定义变量的维度
VarSize = [1 nVar];

% 灰狼数量和最大迭代次数
GreyWolves_num = 100;
MaxIt = 50;  % 最大迭代次数

% 存档大小
Archive_size = 100;   % 存档的大小

% 设置参数
alpha = 0.1;  % 网格膨胀参数
nGrid = 10;   % 每个维度的网格数量
beta = 4;     % 领头选择压力参数
gamma = 2;    % 删除多余成员的选择压力

% 初始化灰狼种群
GreyWolves = CreateEmptyParticle(GreyWolves_num);
for i = 1:GreyWolves_num
    GreyWolves(i).Velocity = 0;  % 初始化速度
    GreyWolves(i).Position = zeros(1, nVar);  % 初始化位置
    for j = 1:nVar
        % 在下界和上界之间均匀随机生成位置
        GreyWolves(i).Position(1, j) = unifrnd(lb(j), ub(j), 1);
    end
    % 计算当前位置的目标函数值
    GreyWolves(i).Cost = fobj(GreyWolves(i).Position')';
    % 初始化最佳位置和目标函数值
    GreyWolves(i).Best.Position = GreyWolves(i).Position;
    GreyWolves(i).Best.Cost = GreyWolves(i).Cost;
end

% 确定种群的支配关系
GreyWolves = DetermineDomination(GreyWolves);

% 获取非支配的个体
Archive = GetNonDominatedParticles(GreyWolves);

% 提取非支配个体的成本
Archive_costs = GetCosts(Archive);

% 创建超立方体
G = CreateHypercubes(Archive_costs, nGrid, alpha);

% 计算每个存档个体的网格索引
for i = 1:numel(Archive)
    [Archive(i).GridIndex, Archive(i).GridSubIndex] = GetGridIndex(Archive(i), G);
end

% MOGWO 主循环
for it = 1:MaxIt
    a = 2 - it * ((2) / MaxIt);  % 动态调整参数 a

    for i = 1:GreyWolves_num
        
        clear rep2
        clear rep3
        
        % 选择 alpha, beta, 和 delta 灰狼
        Delta = SelectLeader(Archive, beta);
        Beta = SelectLeader(Archive, beta);
        Alpha = SelectLeader(Archive, beta);
        
        % 如果在最少拥挤的超立方体中少于三个解，则从第二少拥挤的超立方体中选择其他领导
        if size(Archive, 1) > 1
            counter = 0;
            for newi = 1:size(Archive, 1)
                if sum(Delta.Position ~= Archive(newi).Position) ~= 0
                    counter = counter + 1;
                    rep2(counter, 1) = Archive(newi);
                end
            end
            Beta = SelectLeader(rep2, beta);
        end
        
        % 如果第二少拥挤的超立方体中只有一个解，则从第三少拥挤的超立方体中选择 delta 领导
        if size(Archive, 1) > 2
            counter = 0;
            for newi = 1:size(rep2, 1)
                if sum(Beta.Position ~= rep2(newi).Position) ~= 0
                    counter = counter + 1;
                    rep3(counter, 1) = rep2(newi);
                end
            end
            Alpha = SelectLeader(rep3, beta);
        end
        
        % 计算新的位置
        % Eq.(3.4)
        c = 2 .* rand(1, nVar);
        % Eq.(3.1)
        D = abs(c .* Delta.Position - GreyWolves(i).Position);
        % Eq.(3.3)
        A = 2 .* a .* rand(1, nVar) - a;
        % Eq.(3.8)
        X1 = Delta.Position - A .* abs(D);
        
        % 计算新的位置
        % Eq.(3.4)
        c = 2 .* rand(1, nVar);
        % Eq.(3.1)
        D = abs(c .* Beta.Position - GreyWolves(i).Position);
        % Eq.(3.3)
        A = 2 .* a .* rand(1, nVar) - a;
        % Eq.(3.9)
        X2 = Beta.Position - A .* abs(D);
        
        % 计算新的位置
        % Eq.(3.4)
        c = 2 .* rand(1, nVar);
        % Eq.(3.1)
        D = abs(c .* Alpha.Position - GreyWolves(i).Position);
        % Eq.(3.3)
        A = 2 .* a .* rand(1, nVar) - a;
        % Eq.(3.10)
        X3 = Alpha.Position - A .* abs(D);
        
        % Eq.(3.11)
        GreyWolves(i).Position = (X1 + X2 + X3) ./ 3;
        
        % 边界检查
        GreyWolves(i).Position = min(max(GreyWolves(i).Position, lb), ub);
        
        % 计算当前粒子的目标函数值
        GreyWolves(i).Cost = fobj(GreyWolves(i).Position')';
    end
    
    % 确定支配关系
    GreyWolves = DetermineDomination(GreyWolves);
    
    % 获取非支配的灰狼
    non_dominated_wolves = GetNonDominatedParticles(GreyWolves);
    
    % 更新存档
    Archive = [Archive; non_dominated_wolves];
    
    % 更新存档的支配关系
    Archive = DetermineDomination(Archive);
    
    % 获取非支配的存档
    Archive = GetNonDominatedParticles(Archive);
    
    % 计算每个存档个体的网格索引
    for i = 1:numel(Archive)
        [Archive(i).GridIndex, Archive(i).GridSubIndex] = GetGridIndex(Archive(i), G);
    end
    
    % 如果存档超出了设定大小，则删除多余成员
    if numel(Archive) > Archive_size
        EXTRA = numel(Archive) - Archive_size;
        Archive = DeleteFromRep(Archive, EXTRA, gamma);
        
        % 更新存档成本并重新创建超立方体
        Archive_costs = GetCosts(Archive);
        G = CreateHypercubes(Archive_costs, nGrid, alpha);
    end
    
    % 打印当前迭代的存档解决方案数量
    disp(['在第 ' num2str(it) ' 次迭代中: 存档中的解数量 = ' num2str(numel(Archive))]);
    
    % 保存结果
    save results
    
    % 绘制结果
    costs = GetCosts(GreyWolves);
    Archive_costs = GetCosts(Archive);
    
    if drawing_flag == 1
        hold off
        plot(costs(1,:), costs(2,:), 'k.');  % 绘制灰狼
        hold on
        plot(Archive_costs(1,:), Archive_costs(2,:), 'rd');  % 绘制非支配解决方案
        legend('灰狼', '非支配解决方案');
        drawnow
    end
end
```

### 注释说明：
1. **全局设置**：
   - 清除工作区和命令窗口，设置绘图标志、变量数量和目标函数。

2. **参数设置**：
   - 定义优化过程中使用的边界、灰狼数量、最大迭代次数、存档大小及各种算法参数。

3. **初始化**：
   - 创建灰狼粒子，初始化其位置、速度和目标函数值，并计算每个粒子的最佳位置和目标函数值。

4. **主循环**：
   - 进行最大迭代次数的循环，动态调整参数并选择适应度较高的粒子。
   - 计算新位置时使用多个选择的领导者，以增强搜索的多样性。
   - 进行边界检查以确保粒子在可行域内，并计算新的目标函数值。

5. **更新存档**：
   - 确保存档中只保留非支配解，并根据存档的大小进行调整。

6. **绘图**：
   - 如果绘图标志为1，则在每次迭代中绘制当前灰

狼的位置和存档中的非支配解。

这个文件实现了多目标灰狼优化算法的主要框架，结合了多种选择策略，以提高寻找Pareto前沿的能力。

## `Plot_ZDT1.m` 文件的详细中文注释版本：

```matlab
function z = Plot_ZDT1()
    % 定义目标函数，使用 ZDT3 函数
    ObjectiveFunction = @(x) ZDT3(x);
    
    % 定义 x 的取值范围
    x = 0:0.01:1;  % 从0到1，以0.01为步长
    
    % 计算目标函数的值
    for i = 1:size(x, 2)
        TPF(i, :) = ObjectiveFunction([x(i) 0 0 0 0]);  % 计算目标函数的目标值
    end
    
    % 绘制目标值的曲线
    line(TPF(:, 1), TPF(:, 2));
    
    % 设置图形的标题和坐标轴标签
    title('ZDT1');
    xlabel('f1');
    ylabel('f2');
    box on;  % 添加边框

    % 获取当前图形并设置字体属性
    fig = gcf;
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Garamond');  % 设置字体为Garamond
    set(findall(fig, '-property', 'FontAngle'), 'FontAngle', 'italic');  % 设置字体为斜体
end
```

### 注释说明：
1. **函数定义**：
   - 定义函数 `Plot_ZDT1`，该函数用于绘制 ZDT1 函数的目标值。

2. **目标函数**：
   - 使用匿名函数定义目标函数 `ObjectiveFunction`，该函数以输入向量 `x` 计算 ZDT3 函数的值。

3. **计算目标值**：
   - 在 `x` 的范围内循环计算目标函数值，`TPF` 数组用于存储目标函数的输出值。

4. **绘图**：
   - 使用 `line` 函数绘制目标函数输出的曲线。
   - 设置图形的标题和坐标轴标签，并添加边框以增强可视化效果。

5. **字体设置**：
   - 获取当前图形的句柄，设置字体为 Garamond，并将字体倾斜，以使图形的视觉效果更加美观。

这个文件实现了对 ZDT1 的可视化功能，主要用于展示在多目标优化中该函数的目标值。

## `RouletteWheelSelection.m` 文件的详细中文注释版本：

```matlab
function i = RouletteWheelSelection(p)
    % 轮盘赌选择方法
    % 输入:
    %   p - 概率向量，表示每个个体被选择的概率
    % 输出:
    %   i - 被选择个体的索引

    r = rand;  % 生成一个在 [0, 1] 之间的随机数
    c = cumsum(p);  % 计算概率的累积和

    % 找到第一个累积和大于等于随机数的索引
    i = find(r <= c, 1, 'first');  
end
```

### 注释说明：
1. **函数定义**：
   - 定义函数 `RouletteWheelSelection`，该函数实现了轮盘赌选择的机制。

2. **参数**：
   - 输入参数 `p` 是一个概率向量，表示每个个体被选择的概率。

3. **随机数生成**：
   - 使用 `rand` 函数生成一个在 0 到 1 之间的随机数 `r`。

4. **累积和计算**：
   - 使用 `cumsum` 函数计算概率向量 `p` 的累积和 `c`，使得 `c` 的每个元素代表选择某个个体的阈值。

5. **选择个体**：
   - 使用 `find` 函数查找第一个累积和大于等于随机数 `r` 的索引，并将其赋值给 `i`，表示被选择的个体的索引。

这个函数在多目标优化或遗传算法中常用于选择操作，允许高适应度个体有更大的概率被选中。

## `SelectLeader.m` 文件的详细中文注释版本：

```matlab
function rep_h = SelectLeader(rep, beta)
    % 选择领导者函数
    % 输入:
    %   rep - 存储个体的结构体数组
    %   beta - 选择压力参数（默认值为1）
    % 输出:
    %   rep_h - 选择的领导者个体

    if nargin < 2
        beta = 1;  % 如果未提供beta参数，默认为1
    end

    % 获取占用的单元格及其成员数量
    [occ_cell_index, occ_cell_member_count] = GetOccupiedCells(rep);
    
    % 计算每个单元格的选择概率
    p = occ_cell_member_count.^(-beta);  % 概率与成员数量的负指数成反比
    p = p / sum(p);  % 归一化概率，使其总和为1
    
    % 使用轮盘赌选择法选择一个单元格
    selected_cell_index = occ_cell_index(RouletteWheelSelection(p));
    
    % 获取被选中单元格的所有成员
    GridIndices = [rep.GridIndex];
    selected_cell_members = find(GridIndices == selected_cell_index);
    
    % 随机选择单元格中的一个个体
    n = numel(selected_cell_members);  % 成员数量
    selected_memebr_index = randi([1, n]);  % 随机选择一个索引
    
    h = selected_cell_members(selected_memebr_index);  % 获取选中的个体索引
    
    rep_h = rep(h);  % 返回选择的领导者个体
end
```

### 注释说明：
1. **函数定义**：
   - 定义函数 `SelectLeader`，该函数用于在个体库中选择一个领导者个体。

2. **参数**：
   - 输入参数 `rep` 是一个结构体数组，包含多个个体的信息。
   - 输入参数 `beta` 是选择压力参数，影响选择的随机性（默认值为1）。

3. **获取占用单元**：
   - 使用 `GetOccupiedCells` 函数获取占用的单元格和每个单元格中的成员数量。

4. **计算选择概率**：
   - 根据成员数量计算选择概率，较少成员的单元格有更高的被选中概率。

5. **轮盘赌选择**：
   - 使用 `RouletteWheelSelection` 函数选择一个单元格，依据前面计算的概率。

6. **获取单元格成员**：
   - 找到在选定单元格中的所有个体。

7. **随机选择个体**：
   - 随机选择一个个体作为领导者，并返回该个体。


## `untitled.m` 文件的详细中文注释版本：

```matlab
% 生成一组从0到1的等间距点
t = linspace(0, 1);
% 计算简单多目标函数的值
F = simple_mult(t');
% 绘制目标函数的曲线
plot(t, F', 'LineWidth', 2)
hold on

% 绘制绿色虚线，表示目标函数的约束区域
plot([0, 0], [0, 8], 'g--');
plot([1, 1], [0, 8], 'g--');
% 在图中标记最小值位置
plot([0, 1], [1, 6], 'k.', 'MarkerSize', 15);
text(-0.25, 1.5, 'Minimum(f_1(x))')  % 标注 f1 的最小值位置
text(.75, 5.5, 'Minimum(f_2(x))')    % 标注 f2 的最小值位置
hold off

% 添加图例和标签
legend('f_1(x)', 'f_2(x)')
xlabel({'x'; 'Tradeoff region between the green lines'})

% 使用 fminbnd 找到第一个目标函数的最小值
k = 1;
[min1, minfn1] = fminbnd(@(x)pickindex(x, k), -1, 2);
% 使用 fminbnd 找到第二个目标函数的最小值
k = 2;
[min2, minfn2] = fminbnd(@(x)pickindex(x, k), -1, 2);

goal = [minfn1, minfn2];  % 目标值数组

nf = 2; % 目标函数数量
N = 500; % 用于绘图的点数量
onen = 1/N;  % 每个点的增量
x = zeros(N+1, 1);  % 初始化 x 值
f = zeros(N+1, nf);  % 初始化 f 值
fun = @simple_mult;  % 定义目标函数
x0 = 0.5;  % 初始值
options = optimoptions('fgoalattain', 'Display', 'off');  % 设定目标达成的优化选项

% 对于每个目标函数权重，从 0 到 1 进行循环
for r = 0:N
    t = onen * r; % 当前权重
    weight = [t, 1 - t];  % 权重数组
    % 使用目标达成法求解问题
    [x(r + 1, :), f(r + 1, :)] = fgoalattain(fun, x0, goal, weight, ...
        [], [], [], [], [], [], [], options);
end

% 绘制目标函数值的散点图
figure
plot(f(:, 1), f(:, 2), 'ko');

% 绘制平滑的曲线图
figure
x1 = f(:, 1);  % 第一个目标函数值
y1 = f(:, 2);  % 第二个目标函数值
x2 = linspace(min(x1), max(x1));  % 创建用于插值的 x 值
y2 = interp1(x1, y1, x2, 'spline');  % 使用样条插值平滑曲线
xlabel('f_1')  % x 轴标签
ylabel('f_2')  % y 轴标签
plot(x2, y2);  % 绘制平滑曲线

% 定义简单的多目标函数
function f = simple_mult(x)
    % f(:,1) = sqrt(1+x.^2);  % 第一个目标函数
    % f(:,2) = 4 + 2*sqrt(1+(x-1).^2);  % 第二个目标函数

    n = numel(x);  % 获取 x 的元素个数
    f1 = x(1);  % 第一个目标值
    g = 1 + 9/(n-1) * sum(x(2:end));  % 计算 g 值
    h = 1 - sqrt(f1 / g);  % 计算 h 值
    f2 = g * h;  % 计算第二个目标值
    f = [f1; f2];  % 将目标值组合成列向量返回
end

% 定义索引选择函数
function z = pickindex(x, k)
    z = simple_mult(x);  % 计算目标函数值
    z = z(k);  % 返回第 k 个目标函数值
end
```

### 注释说明：
1. **脚本开头**：设置绘图的初始参数和定义目标函数。
2. **绘制目标函数**：绘制多目标函数的结果，添加标注以说明最小值的位置。
3. **寻找最小值**：使用 `fminbnd` 方法对两个目标函数进行最小化，并记录目标值。
4. **目标达成方法**：利用 `fgoalattain` 函数生成目标函数值的多组解。
5. **绘图**：生成目标函数值的散点图和通过插值生成的平滑曲线图。
6. **定义目标函数**：包含计算目标函数值的函数 `simple_mult` 和选择目标函数的辅助函数 `pickindex`。

## `ZDT1.m` 文件的详细中文注释版本：

```matlab
function z = ZDT1(x)
    % ZDT1 函数计算多目标优化问题中的目标值。
    % 输入:
    %   x - 输入变量的列向量，包含 n 个决策变量
    % 输出:
    %   z - 目标函数值的列向量，包含两个目标函数值 f1 和 f2

    n = numel(x);  % 获取决策变量的数量
    f1 = x(1);  % 第一个目标函数值 f1 取决于第一个决策变量

    % 计算 g 值，g 是所有其他决策变量的加权和
    g = 1 + 9 / (n - 1) * sum(x(2:end));  % 计算 g 的值

    % 计算 h 值，h 是一个基于 f1 和 g 的函数
    h = 1 - sqrt(f1 / g);  % h 的计算

    % 计算第二个目标函数值 f2
    f2 = g * h;  % f2 的值与 g 和 h 相关

    % 返回目标函数值的列向量
    z = [f1; f2];  % 将 f1 和 f2 组合成列向量返回
end
```

### 注释说明：
1. **函数定义**：定义了一个函数 `ZDT1`，用于计算两个目标函数值。
2. **输入和输出**：描述了函数的输入（决策变量的列向量 `x`）和输出（目标函数值的列向量 `z`）。
3. **目标函数计算**：
   - 计算第一个目标函数 `f1`，直接取决于输入的第一个决策变量。
   - 计算辅助变量 `g`，是根据其他决策变量计算得出的加权和。
   - 计算辅助变量 `h`，是一个与 `f1` 和 `g` 相关的值。
   - 最后计算第二个目标函数 `f2`，并将 `f1` 和 `f2` 组合成一个列向量返回。

## `ZDT2.m` 文件的详细中文注释版本：

```matlab
function z = ZDT2(x)
    % ZDT2 函数计算多目标优化问题中的目标值。
    % 输入:
    %   x - 输入变量的列向量，包含 n 个决策变量
    % 输出:
    %   z - 目标函数值的列向量，包含两个目标函数值 f1 和 f2

    n = numel(x);  % 获取决策变量的数量
    f1 = x(1);  % 第一个目标函数值 f1 取决于第一个决策变量

    % 计算 g 值，g 是所有其他决策变量的加权和
    g = 1 + 9 / (n - 1) * sum(x(2:end));  % 计算 g 的值

    % 计算 h 值，h 是一个基于 f1 和 g 的函数
    h = 1 - (f1 / g)^2;  % h 的计算

    % 计算第二个目标函数值 f2
    f2 = g * h;  % f2 的值与 g 和 h 相关

    % 返回目标函数值的列向量
    z = [f1; f2];  % 将 f1 和 f2 组合成列向量返回
end
```

### 注释说明：
1. **函数定义**：定义了一个函数 `ZDT2`，用于计算两个目标函数值。
2. **输入和输出**：描述了函数的输入（决策变量的列向量 `x`）和输出（目标函数值的列向量 `z`）。
3. **目标函数计算**：
   - 计算第一个目标函数 `f1`，直接取决于输入的第一个决策变量。
   - 计算辅助变量 `g`，是根据其他决策变量计算得出的加权和。
   - 计算辅助变量 `h`，是一个与 `f1` 和 `g` 相关的值，采用了平方形式。
   - 最后计算第二个目标函数 `f2`，并将 `f1` 和 `f2` 组合成一个列向量返回。

## `ZDT3.m` 文件的详细中文注释版本：

```matlab
function z = ZDT3(x)
    % ZDT3 函数计算多目标优化问题中的目标值。
    % 输入:
    %   x - 输入变量的列向量，包含 n 个决策变量
    % 输出:
    %   z - 目标函数值的列向量，包含两个目标函数值 f1 和 f2

    n = numel(x);  % 获取决策变量的数量
    f1 = x(1);  % 第一个目标函数值 f1 取决于第一个决策变量

    % 计算 g 值，g 是所有其他决策变量的加权和
    g = 1 + 9 / (n - 1) * sum(x(2:end));  % 计算 g 的值

    % 计算 h 值，h 是一个基于 f1 和 g 的函数，并引入了一个正弦项
    h = 1 - f1 / g - (f1 / g) * sin(10 * pi * x(1));  % h 的计算

    % 计算第二个目标函数值 f2
    f2 = g * h;  % f2 的值与 g 和 h 相关

    % 返回目标函数值的列向量
    z = [f1; f2];  % 将 f1 和 f2 组合成列向量返回
end
```

### 注释说明：
1. **函数定义**：定义了一个函数 `ZDT3`，用于计算两个目标函数值。
2. **输入和输出**：描述了函数的输入（决策变量的列向量 `x`）和输出（目标函数值的列向量 `z`）。
3. **目标函数计算**：
   - 计算第一个目标函数 `f1`，直接取决于输入的第一个决策变量。
   - 计算辅助变量 `g`，是根据其他决策变量计算得出的加权和。
   - 计算辅助变量 `h`，是一个与 `f1` 和 `g` 相关的值，包含一个正弦项以引入非线性特性。
   - 最后计算第二个目标函数 `f2`，并将 `f1` 和 `f2` 组合成一个列向量返回。

## `ZDT4.m` 文件的详细中文注释版本：

```matlab
function z = ZDT4(x)
    % ZDT4 函数计算多目标优化问题中的目标值。
    % 输入:
    %   x - 输入变量的列向量，包含 n 个决策变量
    % 输出:
    %   z - 目标函数值的列向量，包含两个目标函数值 f1 和 f2

    n = numel(x);  % 获取决策变量的数量
    f1 = x(1);  % 第一个目标函数值 f1 取决于第一个决策变量

    % 初始化 sum 为 0，用于计算 g 的值
    sum = 0;  
    % 计算 g 值，g 是根据所有其他决策变量计算得出的
    for i = 2:n
        % 根据公式累加每个决策变量的贡献
        sum = sum + (x(i)^2 - 10 * cos(4 * pi * x(i)));
    end

    % g 的计算，包含固定项和根据决策变量计算的部分
    g = 1 + (n - 1) * 10 + sum;  % g 的值

    % 计算第二个目标函数值 f2
    f2 = g * (1 - (f1 / g)^0.5);  % f2 的值与 g 和 f1 相关

    % 返回目标函数值的列向量
    z = [f1; f2];  % 将 f1 和 f2 组合成列向量返回
end
```

### 注释说明：
1. **函数定义**：定义了一个函数 `ZDT4`，用于计算两个目标函数值。
2. **输入和输出**：描述了函数的输入（决策变量的列向量 `x`）和输出（目标函数值的列向量 `z`）。
3. **目标函数计算**：
   - 计算第一个目标函数 `f1`，直接取决于输入的第一个决策变量。
   - 通过循环计算辅助变量 `g`，该变量是所有其他决策变量的函数，包含余弦项以引入复杂性。
   - 计算第二个目标函数 `f2`，其值与 `g` 和 `f1` 相关。
   - 最后将 `f1` 和 `f2` 组合成一个列向量返回。

## `results.mat` 文件的作用：

在你的 MOGWO（多目标灰狼优化器）实现中，`results.mat` 文件的作用主要是用于保存算法运行过程中生成的数据。这些数据通常包括：

1. **归档结果（Archive）**：存储在每次迭代中非支配解的集合。这个集合包含了当前找到的最优解，通常用于评估算法的性能。

2. **每次迭代的状态**：可以包含在每次迭代中记录的参数，如当前迭代次数、当前归档中的解决方案数量等。

3. **目标函数值**：可以保存当前种群的目标函数值，以便后续分析或可视化。

4. **算法运行的历史数据**：例如，随着迭代次数的增加，记录算法的收敛情况、目标函数的变化等。

保存这些数据的主要目的是为了便于后续分析和结果的可视化。你可以在 MATLAB 中加载这个 `.mat` 文件，查看保存的变量和数据，以便于评估算法的表现、理解其收敛性以及分析结果。

### 具体用法
在代码中使用 `save results` 语句时，会将当前工作区中的所有变量（除非使用了其他参数来限制保存的变量）保存到 `results.mat` 文件中。你可以通过如下命令加载这个文件：

```matlab
load('results.mat');
```

这将会把 `results.mat` 文件中的所有变量加载到当前工作区，你就可以查看和分析这些数据了。

