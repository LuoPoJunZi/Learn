# 非支配性排序遗传算法 II---NSGA-II

非支配性排序遗传算法 II（NSGA-II）是一种用于解决多目标优化问题的进化算法。它在遗传算法的基础上引入了非支配排序和拥挤距离的概念，以有效地找到多个目标函数的最优解集。以下是NSGA-II的主要特点和步骤：

### 1. 主要特点
- **多目标优化**：NSGA-II能够同时优化多个相互冲突的目标函数，例如在设计中可能需要最小化成本和最大化性能。
- **非支配排序**：通过对种群中的个体进行非支配排序，将个体分为不同的等级，以选择出 Pareto 前沿的解。
- **拥挤距离**：为了保持解的多样性，NSGA-II计算每个个体的拥挤距离，作为选择的一个指标，优先选择拥挤距离大的个体。
- **快速非支配排序**：NSGA-II的排序算法能够在较短的时间内处理大规模种群，相比于早期的非支配排序算法具有更高的效率。

### 2. 算法步骤
以下是NSGA-II的基本步骤：

1. **初始化**：
   - 随机生成初始种群，个体由决策变量构成，并计算每个个体的目标函数值。

2. **非支配排序**：
   - 对种群中的个体进行非支配排序，识别出不同等级的解（如Pareto前沿）。

3. **计算拥挤距离**：
   - 为每个个体计算拥挤距离，评估其在前沿中的稀疏程度，帮助维护解的多样性。

4. **选择、交叉和突变**：
   - 通过选择操作（如锦标赛选择）从当前种群中选择父代，执行交叉操作生成子代，并进行突变以引入新的基因。

5. **合并种群**：
   - 将父代和子代合并成新的种群。

6. **再次进行非支配排序和拥挤距离计算**：
   - 对合并后的种群进行非支配排序和拥挤距离计算，准备下一代的选择。

7. **更新种群**：
   - 根据支配等级和拥挤距离选择出下一代的种群，保证种群规模不变。

8. **重复迭代**：
   - 重复执行选择、交叉、突变、合并和排序步骤，直到达到设定的最大迭代次数或满足终止条件。

### 3. 应用领域
NSGA-II广泛应用于多个领域，包括但不限于：
- 工程设计（如结构优化、产品设计）
- 交通流优化
- 资源分配
- 生物信息学
- 经济模型优化

### 4. 优势与挑战
- **优势**：
  - 能够有效处理多目标问题。
  - 提供解的多样性和覆盖面。
  - 算法效率高，适合大规模优化。

- **挑战**：
  - 对于某些问题，解的分布可能不均匀，需要调整参数以优化性能。
  - 对于高维目标问题，算法的复杂性和计算负担可能增加。

### 总结
NSGA-II是一种强大的多目标优化算法，通过非支配排序和拥挤距离机制在保证解质量的同时维护解的多样性，适合于解决复杂的优化问题。

## `CalcCrowdingDistance.m` 文件代码：

```matlab
function pop=CalcCrowdingDistance(pop,F)
    % 计算种群中每个个体的拥挤度，用于NSGA-II算法中的拥挤度比较操作
    
    nF = numel(F);  % 获取非支配前沿的数量
    
    for k = 1:nF  % 遍历每一个非支配前沿
        
        Costs = [pop(F{k}).Cost];  % 获取当前非支配前沿个体的目标函数值
        
        nObj = size(Costs, 1);  % 目标函数的个数
        
        n = numel(F{k});  % 当前非支配前沿中个体的数量
        
        d = zeros(n, nObj);  % 初始化拥挤度矩阵
        
        for j = 1:nObj  % 对每个目标函数计算拥挤度
            
            [cj, so] = sort(Costs(j,:));  % 按照第j个目标函数对个体进行排序
            
            d(so(1), j) = inf;  % 最边界的个体的拥挤度设为无穷大
            
            for i = 2:n-1
                % 计算第i个个体在第j个目标上的拥挤度
                d(so(i), j) = abs(cj(i+1) - cj(i-1)) / abs(cj(1) - cj(end));
                % 公式： (下一邻居目标值 - 上一邻居目标值) / (最大目标值 - 最小目标值)
            end
            
            d(so(end), j) = inf;  % 另一端边界个体的拥挤度设为无穷大
            
        end
        
        % 计算每个个体的总体拥挤度
        for i = 1:n
            pop(F{k}(i)).CrowdingDistance = sum(d(i,:));  % 拥挤度是对各目标上的拥挤度求和
        end
        
    end

end
```

### 注释要点：
1. **拥挤度计算**：通过对每个目标函数进行排序，然后计算每个个体的邻居间的差异，来评估个体在当前目标空间中的“拥挤”程度。
2. **边界处理**：位于目标函数值边界的个体的拥挤度被设为无穷大，以确保边界个体不会被轻易舍弃。
3. **多目标适用性**：算法可以同时处理多个目标，逐个目标计算拥挤度。

## `Crossover.m` 文件代码：

```matlab
function [y1, y2] = Crossover(x1, x2)
    % 执行模拟二进制交叉（SBX）操作，生成两个子代
    % 输入：
    %   x1 - 父代1的解向量
    %   x2 - 父代2的解向量
    % 输出：
    %   y1 - 子代1的解向量
    %   y2 - 子代2的解向量

    alpha = rand(size(x1));  % 随机生成与解向量相同维度的[0,1]之间的随机数矩阵
    
    % 生成两个子代
    y1 = alpha .* x1 + (1 - alpha) .* x2;  % 子代1的解为父代1和父代2按比例混合
    y2 = alpha .* x2 + (1 - alpha) .* x1;  % 子代2的解为父代2和父代1按比例混合
    
end
```

### 注释要点：
1. **交叉操作**：该代码实现了一种简单的线性交叉方法，基于随机系数 `alpha`，生成两个新个体 `y1` 和 `y2`。
2. **随机权重**： `alpha` 的随机性确保了每次交叉生成的子代个体都有不同的基因组合。
3. **适应不同维度**：该代码能够处理不同长度的解向量，适应性较强。

## `Dominates.m` 文件代码：

```matlab
function b = Dominates(x, y)
    % 判断解x是否支配解y，基于Pareto支配的概念
    % 输入：
    %   x - 解向量或包含目标函数值的结构体
    %   y - 解向量或包含目标函数值的结构体
    % 输出：
    %   b - 布尔值，若x支配y则返回true，否则返回false

    if isstruct(x)
        x = x.Cost;  % 如果x是结构体，提取其目标函数值
    end

    if isstruct(y)
        y = y.Cost;  % 如果y是结构体，提取其目标函数值
    end

    % 支配条件：
    % 1. x的所有目标函数值小于等于y的目标函数值（all(x<=y)）
    % 2. 至少有一个目标函数值x严格小于y（any(x<y)）
    b = all(x <= y) && any(x < y);  % 如果两个条件都满足，则x支配y
    
end
```

### 注释要点：
1. **Pareto支配**：判断解 `x` 是否在多目标优化中支配解 `y`，即 `x` 在所有目标上都不差于 `y`，并且至少在一个目标上优于 `y`。
2. **结构体处理**：如果 `x` 或 `y` 是结构体，该函数会提取它们的 `Cost` 属性，这通常代表解的目标函数值。
3. **支配条件**：函数核心是两个条件，分别是目标值的逐项比较和严格不等判断。

## `main.m` 文件非常简洁：

```matlab
% 主程序文件，用于运行NSGA-II算法

% 调用NSGA-II算法的主函数
nsga2;
```

### 注释要点：
1. **主程序**：`main.m` 文件是整个程序的入口，通过调用 `nsga2` 函数来运行非支配性排序遗传算法 II (NSGA-II)。
2. **简单启动**：它仅包含对核心算法函数的调用，具体的算法逻辑将在 `nsga2` 函数中实现。

## `MOP2.m` 文件代码：

```matlab
function z = MOP2(x)
    % MOP2：测试问题的目标函数，用于多目标优化问题
    % 输入：
    %   x - 决策变量向量
    % 输出：
    %   z - 包含两个目标函数值的列向量

    n = numel(x);  % 获取决策变量的个数
    
    % 目标函数1：一个基于平方和的函数，目标是最小化
    z1 = 1 - exp(-sum((x - 1/sqrt(n)).^2));
    
    % 目标函数2：另一个基于平方和的函数，目标是最小化
    z2 = 1 - exp(-sum((x + 1/sqrt(n)).^2));
    
    % 返回两个目标函数值组成的列向量
    z = [z1 z2]';

end
```

### 注释要点：
1. **测试问题**：`MOP2` 是一个标准的多目标优化问题（MOP，Multi-Objective Problem），通常用于测试多目标优化算法的性能。
2. **目标函数**：`z1` 和 `z2` 分别是两个目标函数，都是基于决策变量 `x` 的平方和构造的非线性函数。它们的目标是最小化。
3. **决策变量维度**：目标函数使用决策变量的个数 `n` 来标准化表达式，确保适用于不同维度的输入。

## `MOP4.m` 文件代码：

```matlab
function z = MOP4(x)
    % MOP4：另一个测试问题的目标函数，用于多目标优化问题
    % 输入：
    %   x - 决策变量向量
    % 输出：
    %   z - 包含两个目标函数值的列向量

    a = 0.8;  % 参数a，用于目标函数2
    b = 3;    % 参数b，用于目标函数2
    
    % 目标函数1：基于相邻决策变量的平方和的非线性函数
    z1 = sum(-10 * exp(-0.2 * sqrt(x(1:end-1).^2 + x(2:end).^2)));
    % 对相邻的x(i)和x(i+1)进行组合，并通过指数函数构造一个最小化问题
    
    % 目标函数2：基于决策变量绝对值和正弦函数的组合
    z2 = sum(abs(x).^a + 5 * (sin(x)).^b);
    % 包含x的绝对值部分以及正弦函数的非线性项，目标同样是最小化
    
    % 返回两个目标函数值组成的列向量
    z = [z1 z2]';

end
```

### 注释要点：
1. **目标函数1**：`z1` 是一个非线性函数，依赖于相邻决策变量的平方和，通过指数函数的方式构造，用于创建复杂的搜索空间。
2. **目标函数2**：`z2` 包含决策变量的绝对值以及正弦函数的高次幂，这种组合构造了不同特性的目标函数，增加了多样性。
3. **多目标优化**：`MOP4` 是一个用于测试多目标优化算法的函数，含有两个目标函数，需要同时最小化。

## `Mutate.m` 文件代码：

```matlab
function y = Mutate(x, mu, sigma)
    % 执行突变操作，用于遗传算法中的个体变异
    % 输入：
    %   x - 原始个体的解向量
    %   mu - 突变概率，表示突变的决策变量比例
    %   sigma - 突变强度，控制突变的幅度（标准差）
    % 输出：
    %   y - 突变后的个体解向量

    nVar = numel(x);  % 获取决策变量的数量
    
    nMu = ceil(mu * nVar);  % 根据突变概率mu计算突变变量的数量
    
    % 随机选择nMu个要突变的决策变量
    j = randsample(nVar, nMu);  
    
    % 如果sigma是向量，确保只针对选择的决策变量应用对应的sigma值
    if numel(sigma) > 1
        sigma = sigma(j);
    end
    
    y = x;  % 复制原始个体
    
    % 对选择的决策变量应用高斯噪声进行突变
    y(j) = x(j) + sigma .* randn(size(j));  
    
end
```

### 注释要点：
1. **突变操作**：该函数通过为某些随机选择的决策变量添加高斯噪声，实现遗传算法中的个体突变。突变增加了解空间的多样性，有助于避免局部最优。
2. **突变比例**：`mu` 控制了要突变的决策变量的比例，`nMu` 是实际要突变的变量数量。
3. **突变强度**：`sigma` 控制了突变的强度，函数支持对不同变量使用不同的突变幅度。
4. **高斯噪声**：突变操作通过正态分布随机数 `randn` 生成噪声，调整变量的值。

## `NonDominatedSorting.m` 文件代码：

```matlab
function [pop, F] = NonDominatedSorting(pop)
    % 非支配排序函数，用于NSGA-II算法中的非支配排序操作
    % 输入：
    %   pop - 种群（个体集合），包含个体的目标函数值和支配信息
    % 输出：
    %   pop - 更新后的种群，包含个体的支配集和支配计数信息
    %   F - 各个非支配前沿的个体索引集合

    nPop = numel(pop);  % 种群中个体的数量

    % 初始化每个个体的支配集（DominationSet）和被支配计数（DominatedCount）
    for i = 1:nPop
        pop(i).DominationSet = [];  % 初始化支配集为空
        pop(i).DominatedCount = 0;  % 初始化被支配计数为0
    end
    
    F{1} = [];  % 初始化第一个非支配前沿

    % 进行两两个体比较，确定支配关系
    for i = 1:nPop
        for j = i + 1:nPop
            p = pop(i);
            q = pop(j);
            
            % 判断个体p是否支配个体q
            if Dominates(p, q)
                p.DominationSet = [p.DominationSet j];  % p支配q，q加入p的支配集
                q.DominatedCount = q.DominatedCount + 1;  % q的被支配计数加1
            end
            
            % 判断个体q是否支配个体p
            if Dominates(q.Cost, p.Cost)
                q.DominationSet = [q.DominationSet i];  % q支配p，p加入q的支配集
                p.DominatedCount = p.DominatedCount + 1;  % p的被支配计数加1
            end
            
            pop(i) = p;  % 更新个体p
            pop(j) = q;  % 更新个体q
        end
        
        % 如果个体i没有被其他个体支配，则它属于第一个非支配前沿
        if pop(i).DominatedCount == 0
            F{1} = [F{1} i];  % 将个体i加入第一个非支配前沿
            pop(i).Rank = 1;  % 设置该个体的Rank为1
        end
    end
    
    k = 1;  % 当前非支配前沿的索引

    % 迭代生成后续非支配前沿
    while true
        
        Q = [];  % 临时数组，用于存储当前非支配前沿的个体
        
        % 遍历当前非支配前沿的所有个体
        for i = F{k}
            p = pop(i);
            
            % 遍历p支配的个体
            for j = p.DominationSet
                q = pop(j);
                
                q.DominatedCount = q.DominatedCount - 1;  % 被支配计数减1
                
                % 如果q的被支配计数为0，说明它属于下一非支配前沿
                if q.DominatedCount == 0
                    Q = [Q j];  % 将q加入临时数组Q
                    q.Rank = k + 1;  % 设置q的Rank为当前前沿的下一层
                end
                
                pop(j) = q;  % 更新个体q
            end
        end
        
        % 如果Q为空，说明所有个体已经完成排序，跳出循环
        if isempty(Q)
            break;
        end
        
        F{k + 1} = Q;  % 将Q作为新的非支配前沿
        k = k + 1;  % 递增前沿索引
        
    end

end
```

### 注释要点：
1. **非支配排序**：该函数实现了多目标优化中的非支配排序，确定种群中的个体属于哪个非支配前沿。前沿越靠前的个体被认为越优。
2. **支配关系的确定**：通过两两比较种群中的个体，判断一个个体是否支配另一个个体，支配关系的判断基于 `Dominates` 函数。
3. **非支配前沿生成**：通过迭代，函数不断生成新的非支配前沿，直到种群中的所有个体都被排序。


## `nsga2.m` 文件代码：

```matlab
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
```

### 注释要点：
1. **问题定义**：设置目标函数和决策变量的上下界，定义多目标优化问题。
2. **NSGA-II 参数**：设置遗传算法的相关参数，如种群大小、交叉和突变的概率、突变率等。
3. **初始化**：生成初始种群，计算每个个体的目标函数值，并进行非支配排序和拥挤距离计算。
4. **主循环**：在每一代中，执行交叉、突变、合并种群、非支配排序、拥挤距离计算和种群截断操作。
5. **结果展示**：每次迭代后显示当前最优前沿（F1）的个体数量，并绘制其目标函数值。

## `PlotCosts.m` 文件代码：

```matlab
function PlotCosts(pop)
    % 绘制非支配解的目标函数值
    % 输入：
    %   pop - 种群（个体集合），包含每个个体的目标函数值

    Costs = [pop.Cost];  % 提取种群中所有个体的目标函数值

    % 绘制目标函数值
    plot(Costs(1,:), Costs(2,:), 'r*', 'MarkerSize', 8);
    xlabel('1^{st} Objective');  % X轴标签：第一个目标
    ylabel('2^{nd} Objective');   % Y轴标签：第二个目标
    title('Non-dominated Solutions (F_{1})');  % 图表标题
    grid on;  % 显示网格
end
```

### 注释要点：
1. **函数目的**：该函数用于绘制非支配解的目标函数值，便于可视化优化结果。
2. **输入参数**：`pop` 是包含个体的种群结构，其中每个个体都有目标函数值。
3. **目标函数值提取**：从种群中提取所有个体的目标函数值，以便进行绘图。
4. **绘图**：使用 `plot` 函数绘制目标函数值，设置图标样式为红色星形标记，标记大小为8。
5. **轴标签和标题**：设置X轴和Y轴的标签，并为图表添加标题，以清楚表明图表的含义。
6. **网格显示**：启用网格，使图表更易于阅读。


## `SortPopulation.m` 文件代码：

```matlab
function [pop, F] = SortPopulation(pop)
    % 对种群进行排序
    % 输入：
    %   pop - 种群（个体集合），每个个体包含其拥挤距离和支配等级
    % 输出：
    %   pop - 排序后的种群
    %   F - 每个等级对应的个体索引

    % 基于拥挤距离进行排序
    [~, CDSO] = sort([pop.CrowdingDistance], 'descend');  % 降序排序
    pop = pop(CDSO);  % 按照拥挤距离排序后的顺序更新种群
    
    % 基于支配等级进行排序
    [~, RSO] = sort([pop.Rank]);  % 升序排序
    pop = pop(RSO);  % 按照支配等级排序后的顺序更新种群
    
    % 更新前沿集合 (Update Fronts)
    Ranks = [pop.Rank];  % 获取排序后的个体的支配等级
    MaxRank = max(Ranks);  % 确定最大的支配等级
    F = cell(MaxRank, 1);  % 初始化每个等级对应的个体索引的单元格数组
    
    for r = 1:MaxRank
        F{r} = find(Ranks == r);  % 找到每个等级对应的个体索引
    end
end
```

### 注释要点：
1. **函数目的**：该函数用于对种群进行排序，主要根据个体的拥挤距离和支配等级，以便选择最优个体。
2. **输入参数**：`pop` 是包含个体的种群结构，每个个体有其拥挤距离和支配等级。
3. **基于拥挤距离排序**：使用 `sort` 函数对个体的拥挤距离进行降序排序，并根据排序结果更新种群。
4. **基于支配等级排序**：对更新后的种群根据支配等级进行升序排序，进一步筛选出最优个体。
5. **更新前沿集合**：通过遍历支配等级，构建一个单元格数组 `F`，其中每个元素包含相同支配等级的个体索引，表示不同等级的个体集合。














