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
