##  `AssociateToReferencePoint.m` 文件代码:

```matlab
function [pop, d, rho] = AssociateToReferencePoint(pop, params)
    % AssociateToReferencePoint 函数的作用是将种群中的个体与参考点关联起来，
    % 并计算每个个体到关联参考点的距离，以及每个参考点关联的个体数量。

    % 输入：
    %   pop - 种群（包含多个个体，每个个体有其对应的目标函数值）
    %   params - 参数结构体，包含参考点 Zr 和参考点的数量 nZr
    
    % 输出：
    %   pop - 更新后的种群，包含每个个体的关联参考点和到参考点的距离
    %   d - 每个个体到所有参考点的距离矩阵
    %   rho - 每个参考点关联的个体数量

    Zr = params.Zr;   % 参考点矩阵，每一列是一个参考点
    nZr = params.nZr; % 参考点的数量
    
    % 初始化 rho，记录每个参考点关联的个体数量
    rho = zeros(1,nZr);
    
    % 初始化距离矩阵 d，记录每个个体到各个参考点的距离
    d = zeros(numel(pop), nZr);
    
    % 遍历种群中的每个个体
    for i = 1:numel(pop)
        % 遍历每个参考点
        for j = 1:nZr
            % 归一化参考点方向向量 w
            w = Zr(:,j)/norm(Zr(:,j));
            % 获取当前个体的归一化目标值 z
            z = pop(i).NormalizedCost;
            % 计算个体到当前参考点的距离（即垂直距离）
            d(i,j) = norm(z - w'*z*w);
        end
        
        % 找到距离最小的参考点，即个体关联的参考点
        [dmin, jmin] = min(d(i,:));
        
        % 记录个体关联的参考点索引
        pop(i).AssociatedRef = jmin;
        % 记录个体到关联参考点的最小距离
        pop(i).DistanceToAssociatedRef = dmin;
        % 增加该参考点的关联个体数量
        rho(jmin) = rho(jmin) + 1;
    end

end
```

### 注释说明：
- `Zr` 是参考点矩阵，每一列代表一个参考点。
- `rho` 是每个参考点关联的个体数量，初始化为零。
- `d` 是个体到参考点的距离矩阵，每一行表示一个个体到所有参考点的距离。
- 通过双重循环，程序遍历种群中的每个个体，并计算其到所有参考点的距离，找到距离最小的参考点作为该个体的关联参考点。
- 最终返回更新后的种群、距离矩阵和关联个体数量。

##  `Crossover.m` 文件代码：

```matlab
function [y1, y2] = Crossover(x1, x2)
    % Crossover 函数执行模拟二进制交叉操作（SBX），生成两个子代个体
    % 输入：
    %   x1, x2 - 父代个体的决策变量向量
    % 输出：
    %   y1, y2 - 子代个体的决策变量向量

    alpha = rand(size(x1));  % 生成与决策变量尺寸相同的随机权重系数 alpha
    
    % 计算第一个子代 y1，公式：y1 = alpha * x1 + (1 - alpha) * x2
    y1 = alpha .* x1 + (1 - alpha) .* x2;
    
    % 计算第二个子代 y2，公式：y2 = alpha * x2 + (1 - alpha) * x1
    y2 = alpha .* x2 + (1 - alpha) .* x1;
    
end
```

### 注释解释：
- **alpha** 是一个随机生成的与输入变量 `x1` 尺寸相同的系数矩阵，用于在两个父代个体之间随机插值。
- **y1** 和 **y2** 是两个子代个体，通过线性插值生成的，这种方式通常用于多目标优化算法中的模拟二进制交叉（SBX）操作。

这种交叉操作的优点是能够产生多样化的子代，有助于算法更好地探索解空间。

## `Dominates.m` 文件代码：

```matlab
function b = Dominates(x, y)
    % Dominates 函数用于检查一个解是否支配另一个解
    % 输入：
    %   x, y - 两个解（可以是包含目标值的结构体或目标值向量）
    % 输出：
    %   b - 布尔值，若 x 支配 y 则为 true，否则为 false

    % 如果 x 是结构体，则提取其目标值
    if isstruct(x)
        x = x.Cost;
    end

    % 如果 y 是结构体，则提取其目标值
    if isstruct(y)
        y = y.Cost;
    end

    % 检查 x 是否支配 y
    % x 支配 y 的条件是：
    % 1. x 的所有目标值都小于或等于 y 的目标值
    % 2. x 至少有一个目标值严格小于 y 的目标值
    b = all(x <= y) && any(x < y);
end
```

### 注释解释：
- **支配定义**：在多目标优化中，解 `x` 被认为支配解 `y`，如果并且仅如果 `x` 在所有目标上都不劣于 `y`，并且在至少一个目标上优于 `y`。这意味着 `x` 要么和 `y` 在所有目标上相等，要么至少有一个目标更好。
- **结构体处理**：如果输入 `x` 或 `y` 是结构体，则提取其 `Cost` 字段，即目标值向量。目标值是算法中用来衡量解优劣的标准。
- **逻辑判断**：`all(x <= y)` 检查 `x` 是否在所有目标上不劣于 `y`，`any(x < y)` 检查是否有至少一个目标上 `x` 优于 `y`。

这种支配关系在进化算法中用于确定哪些个体能够进入下一代，并影响种群的非支配排序。

## `GenerateReferencePoints.m` 文件代码：

```matlab
function Zr = GenerateReferencePoints(M, p)
    % GenerateReferencePoints 函数生成用于多目标优化中的参考点
    % 输入：
    %   M - 目标函数的数量
    %   p - 参考点划分数，控制参考点的密度
    % 输出：
    %   Zr - 生成的参考点矩阵，每列代表一个参考点

    % 调用 GetFixedRowSumIntegerMatrix 函数生成一个元素和固定的整数矩阵
    % 并对结果进行转置和归一化，得到最终的参考点矩阵 Zr
    Zr = GetFixedRowSumIntegerMatrix(M, p)' / p;
end

function A = GetFixedRowSumIntegerMatrix(M, RowSum)
    % GetFixedRowSumIntegerMatrix 函数递归生成一个矩阵，
    % 其中每一行的元素之和为指定的 RowSum
    % 输入：
    %   M - 行的维数（通常是目标函数的数量）
    %   RowSum - 行元素之和
    % 输出：
    %   A - 生成的整数矩阵，每行元素之和固定为 RowSum

    % 检查 M 是否有效，M 必须为正整数
    if M < 1
        error('M cannot be less than 1.');  % 抛出错误：M 不能小于 1
    end
    
    % 检查 M 是否为整数
    if floor(M) ~= M
        error('M must be an integer.');  % 抛出错误：M 必须是整数
    end
    
    % 基本情况：当 M 为 1 时，返回 RowSum
    if M == 1
        A = RowSum;
        return;
    end

    % 初始化 A 为一个空矩阵
    A = [];
    
    % 递归调用，生成所有可能的行组合，使得每行元素之和为 RowSum
    for i = 0:RowSum
        % 对 M-1 维进行递归，RowSum 减去当前的 i 值
        B = GetFixedRowSumIntegerMatrix(M - 1, RowSum - i);
        
        % 将当前的 i 值扩展到对应矩阵的列前，构成新的一行
        A = [A; i * ones(size(B, 1), 1), B];
    end
end
```

### 注释解释：
1. **GenerateReferencePoints** 函数：
   - 该函数用于生成 NSGA-III 算法中的参考点，通过调用递归函数生成一个固定和的整数矩阵，并进行归一化处理，得到最终的参考点矩阵。
   
2. **GetFixedRowSumIntegerMatrix** 函数：
   - 该函数递归生成一个整数矩阵，其中每行元素的和等于给定的 `RowSum`。
   - 这是通过递归的方式，从 `M-1` 维开始生成，再依次向高维扩展，确保每行的元素和固定。
   
3. **应用场景**：
   - 在多目标优化问题中，生成参考点有助于划分解空间，使得解可以在目标空间上均匀分布，从而提高解的多样性。

## `main.m` 文件代码：

```matlab
% main.m
% 主文件，用于启动 NSGA-III 算法的执行

% 调用 nsga3 函数，运行多目标优化算法
nsga3;
```

### 注释解释：
- `main.m` 文件的作用是作为入口点，简单地调用 `nsga3` 函数执行主算法。这个文件通常是为了方便用户启动整个算法。
- 在实际运行中，用户只需要运行 `main.m` 文件，所有的初始化和迭代操作将会在 `nsga3.m` 文件中完成。

## `MOP2.m` 文件代码：

```matlab
function z = MOP2(x)
    % MOP2 函数用于计算多目标优化问题中的两个目标函数值
    % 输入：
    %   x - 决策变量向量
    % 输出：
    %   z - 目标函数值的向量 [z1; z2]

    % 获取决策变量的数量
    n = numel(x);

    % 计算第一个目标函数值 z1
    % 该目标函数表示一个典型的鞍型函数，用指数函数和平方和表示
    z1 = 1 - exp(-sum((x - 1/sqrt(n)).^2));

    % 计算第二个目标函数值 z2
    % 该目标函数与 z1 类似，唯一的区别是平方项中的加减号相反
    z2 = 1 - exp(-sum((x + 1/sqrt(n)).^2));

    % 将 z1 和 z2 作为列向量返回
    z = [z1; z2];
end
```

### 注释解释：
1. **函数 MOP2**:
   - 该函数实现了一个多目标优化问题 (MOP)，定义了两个目标函数 `z1` 和 `z2`。这些目标函数通常用于测试多目标优化算法的性能。
   
2. **z1 和 z2 的计算**:
   - 目标函数 `z1` 和 `z2` 都是指数函数与平方和的组合。
   - `z1` 通过计算 `x` 和常数 `1/sqrt(n)` 之间的差值平方和来确定，`z2` 则计算 `x` 和 `-1/sqrt(n)` 之间的平方和。两者的计算结构类似，但方向不同，形成了互补的优化目标。

3. **返回值**:
   - 返回的 `z` 是一个列向量，包含了两个目标函数值 `z1` 和 `z2`，用于在 NSGA-III 算法中进行优化。

## `Mutate.m` 文件代码：

```matlab
function y = Mutate(x, mu, sigma)
    % Mutate 函数用于执行突变操作，通过随机扰动来改变个体的基因
    % 输入：
    %   x - 决策变量向量（即个体）
    %   mu - 突变率，表示将有多少比例的基因发生突变
    %   sigma - 突变强度，表示突变时的标准差（变化量）
    % 输出：
    %   y - 突变后的个体

    % 获取决策变量的数量
    nVar = numel(x);
    
    % 根据突变率 mu 计算需要突变的基因数量 nMu
    nMu = ceil(mu * nVar);

    % 随机选择 nMu 个基因的位置 j 进行突变
    j = randsample(nVar, nMu);
    
    % 将原始个体 x 赋值给 y，确保 y 基本结构不变
    y = x;
    
    % 在随机选定的基因位置上，进行高斯分布的随机扰动
    % 使用 sigma 控制突变强度，randn 生成正态分布的随机数
    y(j) = x(j) + sigma * randn(size(j));

end
```

### 注释解释：
1. **Mutate 函数**:
   - 该函数实现了遗传算法中的突变操作，通过对个体（决策变量向量）进行随机扰动，探索新的解空间。这有助于防止算法陷入局部最优解。
   
2. **参数解释**:
   - `mu` 是突变率，表示有多少比例的基因会发生突变。根据这个比例，函数会计算需要突变的基因数量 `nMu`。
   - `sigma` 是突变的标准差，控制突变强度。突变是通过在基因位置上加入正态分布的随机扰动实现的。

3. **突变过程**:
   - 首先，函数随机选择若干基因位置（即决策变量中的一些元素），然后对这些位置的基因值进行随机改变。突变的幅度由 `sigma` 设定，并使用正态分布来生成突变量。

## `NonDominatedSorting.m` 文件代码：

```matlab
function [pop, F] = NonDominatedSorting(pop)
    % NonDominatedSorting 函数用于对种群进行非支配排序
    % 输入：
    %   pop - 当前种群（个体集合）
    % 输出：
    %   pop - 更新后的种群，每个个体包含其支配关系信息
    %   F - 包含每个等级（前沿）的个体索引，F{1} 是第一前沿，F{2} 是第二前沿，以此类推

    nPop = numel(pop);  % 获取种群中个体的数量

    % 初始化每个个体的支配集和被支配计数
    for i = 1:nPop
        pop(i).DominationSet = [];  % 该个体支配的个体集合
        pop(i).DominatedCount = 0;   % 该个体被支配的数量
    end
    
    F{1} = [];  % 第一前沿初始化为空
    
    % 对每对个体进行比较以建立支配关系
    for i = 1:nPop
        for j = i + 1:nPop
            p = pop(i);  % 当前个体 p
            q = pop(j);  % 当前个体 q
            
            % 如果 p 支配 q
            if Dominates(p, q)
                p.DominationSet = [p.DominationSet j];  % p 支配的个体集合增加 q
                q.DominatedCount = q.DominatedCount + 1;  % q 被支配计数加 1
            end
            
            % 如果 q 支配 p
            if Dominates(q.Cost, p.Cost)
                q.DominationSet = [q.DominationSet i];  % q 支配的个体集合增加 p
                p.DominatedCount = p.DominatedCount + 1;  % p 被支配计数加 1
            end
            
            pop(i) = p;  % 更新个体 p
            pop(j) = q;  % 更新个体 q
        end
        
        % 如果个体 p 没有被其他个体支配，则将其加入第一前沿
        if pop(i).DominatedCount == 0
            F{1} = [F{1} i];  % 将个体 i 加入第一前沿
            pop(i).Rank = 1;  % 设置个体的等级为 1
        end
    end
    
    k = 1;  % 计数器初始化为 1
    
    while true
        Q = [];  % 当前处理的前沿的个体集合
        
        % 遍历当前前沿 F{k}
        for i = F{k}
            p = pop(i);  % 当前前沿的个体 p
            
            % 遍历 p 的支配集中的每个个体
            for j = p.DominationSet
                q = pop(j);  % 被 p 支配的个体 q
                
                q.DominatedCount = q.DominatedCount - 1;  % 被支配计数减 1
                
                % 如果 q 被支配计数为 0，说明 q 进入下一前沿
                if q.DominatedCount == 0
                    Q = [Q j];  % 将个体 j 加入下一个前沿
                    q.Rank = k + 1;  % 设置个体的等级为 k + 1
                end
                
                pop(j) = q;  % 更新个体 q
            end
        end
        
        % 如果当前前沿 Q 为空，结束循环
        if isempty(Q)
            break;
        end
        
        F{k + 1} = Q;  % 将 Q 加入下一个前沿
        k = k + 1;  % 前沿计数器加 1
    end
end
```

### 注释解释：
1. **NonDominatedSorting 函数**:
   - 该函数实现了非支配排序，用于将种群中的个体分组到不同的等级（前沿），以便在多目标优化中选择最优解。

2. **参数说明**:
   - `pop` 是输入种群，包含多个个体。
   - `F` 是输出变量，保存各个前沿的个体索引。

3. **支配关系的建立**:
   - 每个个体的 `DominationSet` 用于存储其支配的个体，而 `DominatedCount` 用于记录被支配的个体数量。
   - 使用 `Dominates` 函数比较两个个体，确定它们之间的支配关系。

4. **前沿的构建**:
   - 首先将无支配关系的个体（即被支配计数为0的个体）加入第一前沿。
   - 然后通过循环更新每个前沿，直到没有更多的个体可以被加入新的前沿。

## `NormalizePopulation.m` 文件代码：

```matlab
function [pop, params] = NormalizePopulation(pop, params)
    % NormalizePopulation 函数用于对种群进行归一化处理
    % 输入：
    %   pop - 当前种群（个体集合）
    %   params - 包含优化参数和状态的结构体
    % 输出：
    %   pop - 更新后的种群，包含归一化后的成本
    %   params - 更新后的参数结构体

    % 更新理想点 zmin
    params.zmin = UpdateIdealPoint(pop, params.zmin);
    
    % 计算每个个体的成本相对于理想点的偏差
    fp = [pop.Cost] - repmat(params.zmin, 1, numel(pop));
    
    % 使用标量化方法更新参数
    params = PerformScalarizing(fp, params);
    
    % 计算超平面的截距
    a = FindHyperplaneIntercepts(params.zmax);
    
    % 对每个个体进行归一化处理
    for i = 1:numel(pop)
        pop(i).NormalizedCost = fp(:,i) ./ a;  % 归一化成本
    end
end

function a = FindHyperplaneIntercepts(zmax)
    % FindHyperplaneIntercepts 函数用于计算超平面的截距
    % 输入：
    %   zmax - 当前种群中每个目标的最大成本
    % 输出：
    %   a - 超平面的截距
    
    w = ones(1, size(zmax, 2)) / zmax;  % 计算每个目标的权重
    a = (1 ./ w)';  % 计算超平面的截距
end
```

### 注释解释：
1. **NormalizePopulation 函数**:
   - 该函数负责对种群进行归一化处理，使得个体的成本值在同一尺度上进行比较。

2. **参数说明**:
   - `pop` 是输入种群，包含多个个体。
   - `params` 是包含优化参数和状态的结构体，包含理想点（`zmin`）和其他参数。

3. **理想点的更新**:
   - `UpdateIdealPoint` 函数用于根据当前种群更新理想点 `zmin`。

4. **成本的偏差计算**:
   - 计算每个个体的成本相对于理想点的偏差，生成矩阵 `fp`。

5. **标量化方法的应用**:
   - `PerformScalarizing` 函数用于更新标量化参数，确保多目标优化中的权衡。

6. **超平面截距的计算**:
   - `FindHyperplaneIntercepts` 函数用于计算每个目标的超平面截距，生成向量 `a`。

7. **归一化处理**:
   - 将每个个体的成本进行归一化，存储在 `NormalizedCost` 字段中，以便后续比较和选择。

### 使用场景：
这个函数通常用于多目标优化中的种群更新阶段，确保所有个体的目标值在相同的尺度上进行比较，便于选择最优解。

## `nsga3.m` 文件代码：

```matlab
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
```

### 注释解释：
1. **代码开头注释**:
   - 提供了算法的参考文献，描述了 NSGA-III 的背景。

2. **问题定义部分**:
   - 定义了优化问题，包括目标函数、决策变量的数量、变量的范围等。

3. **NSGA-III 参数设置**:
   - 配置了 NSGA-III 算法所需的参数，包括生成的参考点、最大迭代次数、种群大小、交叉和变异的概率等。

4. **收集参数**:
   - 将一些算法参数收集到 `params` 结构体中，便于后续使用。

5. **初始化部分**:
   - 初始化种群，包括生成随机个体和计算它们的成本。

6. **NSGA-III 主循环**:
   - 进行算法的主要迭代过程，包括交叉、变异、合并种群、排序和选择等步骤。

7. **结果输出**:
   - 显示最终的迭代结果和优化结束信息。

### 使用场景：
该代码实现了 NSGA-III 算法用于多目标优化问题，适用于希望同时优化多个目标并处理约束的复杂问题。通过迭代更新种群，寻找 Pareto 前沿解集，为决策者提供多样化的解决方案。

## `PerformScalarizing.m` 文件代码：

```matlab
function params = PerformScalarizing(z, params)

    nObj = size(z, 1);  % 目标函数数量
    nPop = size(z, 2);  % 种群大小
    
    % 如果 smin 参数不为空，使用已有的 zmax 和 smin
    if ~isempty(params.smin)
        zmax = params.zmax;  % 记录当前的最优点
        smin = params.smin;  % 记录当前的最小标量值
    else
        zmax = zeros(nObj, nObj);  % 初始化最优点为零
        smin = inf(1, nObj);  % 初始化最小标量值为无穷大
    end
    
    % 对每个目标函数进行标量化处理
    for j = 1:nObj
       
        w = GetScalarizingVector(nObj, j);  % 获取标量化向量
        
        s = zeros(1, nPop);  % 初始化标量值数组
        for i = 1:nPop
            % 计算每个个体的标量值
            s(i) = max(z(:, i) ./ w);
        end

        [sminj, ind] = min(s);  % 找到当前标量值中的最小值及其索引
        
        % 如果当前最小标量值小于记录的最小值，则更新 zmax 和 smin
        if sminj < smin(j)
            zmax(:, j) = z(:, ind);  % 更新最优点
            smin(j) = sminj;  % 更新最小标量值
        end
        
    end
    
    % 将更新后的 zmax 和 smin 存回 params 结构体中
    params.zmax = zmax;
    params.smin = smin;
    
end

function w = GetScalarizingVector(nObj, j)

    epsilon = 1e-10;  % 设定一个很小的值，以避免除零

    w = epsilon * ones(nObj, 1);  % 初始化标量化向量

    w(j) = 1;  % 将第 j 个目标的权重设置为 1

end
```

### 注释解释：
1. **函数 `PerformScalarizing`**:
   - 该函数的主要功能是对多目标优化中的目标函数值进行标量化处理，以便于后续的优化过程。
   - `z` 是一个矩阵，表示种群中每个个体在每个目标上的值。
   - `params` 是一个结构体，用于存储优化过程中的参数和状态。

2. **参数初始化**:
   - `nObj` 和 `nPop` 分别表示目标函数的数量和种群的大小。
   - `zmax` 用于存储当前已知的最优目标点，`smin` 用于记录每个目标的最小标量值。

3. **标量化处理**:
   - 对于每个目标函数，使用 `GetScalarizingVector` 函数生成一个权重向量。
   - 计算每个个体的标量值，标量值表示该个体在当前目标下的性能。
   - 更新最优目标点和最小标量值。

4. **辅助函数 `GetScalarizingVector`**:
   - 该函数生成一个标量化向量，指定第 j 个目标的权重为 1，其余目标的权重为非常小的值 `epsilon`，以避免除零错误。

### 使用场景：
该代码在多目标优化算法中用于处理目标函数值的标量化，使得不同目标函数的值能够在同一标准下进行比较。这种处理方式在如 NSGA-III 等算法中是必要的，因其可以有效引导搜索过程找到 Pareto 前沿解。

## `PlotCosts.m` 文件代码：

```matlab
function PlotCosts(pop)

    % 从种群中提取成本（目标函数值）
    Costs = [pop.Cost];  % 将每个个体的成本值组合成一个矩阵
    
    % 绘制目标函数值的散点图
    plot(Costs(1, :), Costs(2, :), 'r*', 'MarkerSize', 8);  % 以红色星形标记绘制
    xlabel('1st Objective');  % x 轴标签
    ylabel('2nd Objective');  % y 轴标签
    grid on;  % 开启网格
    
end
```

### 注释解释：
1. **函数 `PlotCosts`**:
   - 该函数用于绘制种群中个体的成本（目标函数值），通常在多目标优化问题中用来可视化目标函数值的分布情况。

2. **参数 `pop`**:
   - `pop` 是一个结构数组，包含了优化过程中所有个体的信息，包括其对应的目标函数值。

3. **提取成本值**:
   - `Costs = [pop.Cost];` 这一行代码将种群中每个个体的目标函数值提取出来，并组合成一个矩阵。矩阵的每一列代表一个个体的目标函数值。

4. **绘图**:
   - `plot(Costs(1, :), Costs(2, :), 'r*', 'MarkerSize', 8);` 这行代码绘制了一个散点图，x 轴为第一个目标的值，y 轴为第二个目标的值，使用红色星形标记，标记大小为 8。
   - `xlabel` 和 `ylabel` 用于设置 x 轴和 y 轴的标签，帮助理解图中的数据。
   - `grid on;` 用于开启图形的网格，使得图形的可读性更高。

### 使用场景：
该函数通常在多目标优化算法的运行过程中被调用，以可视化当前种群中个体在不同目标上的表现，从而观察优化进程和结果的分布情况。这有助于分析算法的性能和解的多样性。

## `SortAndSelectPopulation.m` 文件代码：

```matlab
function [pop, F, params] = SortAndSelectPopulation(pop, params)

    % 归一化种群
    [pop, params] = NormalizePopulation(pop, params);

    % 非支配排序
    [pop, F] = NonDominatedSorting(pop);
    
    nPop = params.nPop;  % 获取种群大小
    if numel(pop) == nPop
        return;  % 如果当前种群数量等于预设数量，则直接返回
    end
    
    % 关联参考点
    [pop, d, rho] = AssociateToReferencePoint(pop, params);
    
    newpop = [];  % 初始化新种群
    for l = 1:numel(F)
        % 如果当前新种群加上当前前沿超出种群大小
        if numel(newpop) + numel(F{l}) > nPop
            LastFront = F{l};  % 记录当前前沿
            break;  % 退出循环
        end
        
        newpop = [newpop; pop(F{l})];   %#ok 添加当前前沿的个体到新种群
    end
    
    while true
        % 找到最小 rho 值的索引 j
        [~, j] = min(rho);
        
        AssocitedFromLastFront = [];  % 记录从最后一个前沿关联的个体
        for i = LastFront
            if pop(i).AssociatedRef == j
                AssocitedFromLastFront = [AssocitedFromLastFront i]; %#ok
            end
        end
        
        % 如果没有与 j 关联的个体，设置 rho(j) 为无穷大
        if isempty(AssocitedFromLastFront)
            rho(j) = inf;
            continue;  % 继续下一个循环
        end
        
        % 如果 rho(j) 为 0，选择距离最近的个体
        if rho(j) == 0
            ddj = d(AssocitedFromLastFront, j);
            [~, new_member_ind] = min(ddj);  % 找到最小距离的个体
        else
            % 否则随机选择一个个体
            new_member_ind = randi(numel(AssocitedFromLastFront));
        end
        
        MemberToAdd = AssocitedFromLastFront(new_member_ind);  % 选择要添加的个体
        
        % 从最后前沿中移除选择的个体
        LastFront(LastFront == MemberToAdd) = [];
        
        newpop = [newpop; pop(MemberToAdd)]; %#ok 添加个体到新种群
        
        rho(j) = rho(j) + 1;  % 更新 rho
        
        % 如果新种群达到预设大小，则退出循环
        if numel(newpop) >= nPop
            break;
        end
    end
    
    % 对新种群进行非支配排序
    [pop, F] = NonDominatedSorting(newpop);
    
end
```

### 注释解释：
1. **函数 `SortAndSelectPopulation`**:
   - 该函数负责对当前种群进行归一化处理、非支配排序和选择操作，以生成新的种群。

2. **归一化种群**:
   - 调用 `NormalizePopulation` 函数对种群进行归一化，以便后续的处理。

3. **非支配排序**:
   - 使用 `NonDominatedSorting` 函数对种群进行排序，并返回排序后的种群和前沿集合 `F`。

4. **种群数量判断**:
   - 如果当前种群数量已达到预设数量，则直接返回。

5. **关联参考点**:
   - 调用 `AssociateToReferencePoint` 函数获取个体与参考点的关联信息，包括距离和关联数量。

6. **生成新种群**:
   - 遍历每个前沿 `F`，将其个体添加到新种群 `newpop`，直到达到预设的种群大小 `nPop`。

7. **处理最后一个前沿**:
   - 在循环中，选择与当前最小 `rho` 值关联的个体添加到新种群中。如果该值为 0，选择距离最近的个体；否则随机选择。

8. **更新种群**:
   - 每次添加个体后更新 `rho` 值，直到新种群达到指定大小。

9. **最终非支配排序**:
   - 对新种群进行非支配排序，返回最终的种群和前沿集合。

### 使用场景：
该函数在多目标进化算法中起到关键作用，通过对种群的排序和选择，确保在每一代中保留优秀的解，促进多样性和收敛性，以便于找到更优的多目标解。

## `UpdateIdealPoint.m` 文件代码：

```matlab
function zmin = UpdateIdealPoint(pop, prev_zmin)
    % 更新理想点（最优解点）
    % 输入：
    %   pop - 当前种群
    %   prev_zmin - 上一轮的理想点
    % 输出：
    %   zmin - 更新后的理想点

    % 如果没有提供 prev_zmin 或其为空，则初始化为无穷大
    if ~exist('prev_zmin', 'var') || isempty(prev_zmin)
        prev_zmin = inf(size(pop(1).Cost));  % 设置为与成本大小相同的无穷大
    end
    
    zmin = prev_zmin;  % 初始化理想点为前一轮的理想点
    for i = 1:numel(pop)
        % 在当前个体成本与当前理想点之间取最小值，更新理想点
        zmin = min(zmin, pop(i).Cost);
    end

end
```

### 注释解释：
1. **函数 `UpdateIdealPoint`**:
   - 该函数用于更新多目标优化中的理想点（最优解点），通常在每一代进化过程中更新。

2. **输入参数**:
   - `pop`: 当前的种群，包含多个个体，每个个体都有对应的成本（目标函数值）。
   - `prev_zmin`: 上一代的理想点。如果没有提供，函数会将其初始化为无穷大。

3. **理想点初始化**:
   - 如果 `prev_zmin` 没有被提供或为空，函数会将其初始化为与个体成本数组相同大小的无穷大数组。

4. **更新理想点**:
   - 函数遍历种群中的每个个体，通过与当前理想点进行比较，更新理想点为当前个体成本与理想点之间的最小值。

### 使用场景：
在多目标进化算法（如 NSGA-II 和 NSGA-III）中，理想点用于跟踪当前已知的最优目标函数值，有助于引导进化过程，确保解的多样性和收敛性。每当新一代的个体生成时，都会调用此函数更新理想点，以反映当前最优解。

