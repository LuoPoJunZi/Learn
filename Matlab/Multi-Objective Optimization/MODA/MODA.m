% MODA.m - 多目标蜻蜓算法 (MODA) 主程序
%
% 该脚本实现了多目标蜻蜓算法，用于求解多目标优化问题。
% 以 ZDT1 测试函数为例，演示了 MODA 的运行过程和结果可视化。

clc;        % 清除命令行窗口
clear;      % 清除工作区变量
close all;  % 关闭所有打开的图形窗口

% %% 根据具体问题修改以下参数 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ObjectiveFunction = @ZDT1;  % 目标函数句柄，这里使用 ZDT1 问题
dim = 5;                    % 决策变量的维度
lb = 0;                     % 决策变量的下界（可以是标量或向量）
ub = 1;                     % 决策变量的上界（可以是标量或向量）
obj_no = 2;                 % 目标函数的数量（ZDT1 为双目标问题）

% 如果上界和下界是标量，则扩展为与维度相同的向量
if size(ub, 2) == 1
    ub = ones(1, dim) * ub;
    lb = ones(1, dim) * lb;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %% MODA 算法的初始参数设定 %%
max_iter = 100;          % 最大迭代次数
N = 100;                 % 蜻蜓群体的数量
ArchiveMaxSize = 100;    % 存档的最大容量

% 初始化存档中的决策变量和目标函数值
Archive_X = zeros(100, dim);          % 存档中解的决策变量（初始为空）
Archive_F = ones(100, obj_no) * inf;  % 存档中解的目标函数值（初始为无穷大）

Archive_member_no = 0;  % 存档中当前成员的数量

% 初始化其他参数
r = (ub - lb) / 2;                     % 邻域半径
V_max = (ub(1) - lb(1)) / 10;         % 最大速度限制（假设所有维度相同）

% 初始化食物和敌人的适应度及位置
Food_fitness = inf * ones(1, obj_no);  % 食物的适应度（初始为无穷大）
Food_pos = zeros(dim, 1);               % 食物的位置

Enemy_fitness = -inf * ones(1, obj_no); % 敌人的适应度（初始为-无穷大）
Enemy_pos = zeros(dim, 1);              % 敌人的位置

% 初始化蜻蜓群体的位置
X = initialization(N, dim, ub, lb);     % 调用 initialization 函数生成初始种群
fitness = zeros(N, 2);                   % 存储每个蜻蜓的适应度

% 初始化速度（DeltaX）
DeltaX = initialization(N, dim, ub, lb);
iter = 0;                                % 迭代计数器

% 记录每个蜻蜓在每次迭代中的位置历史（可选）
position_history = zeros(N, max_iter, dim);

% %% 主循环开始 %%
for iter = 1:max_iter
    % 动态调整邻域半径 r 和权重参数
    r = (ub - lb) / 4 + ((ub - lb) * (iter / max_iter) * 2);
    
    % 线性递减惯性权重 w，从0.9递减到0.2
    w = 0.9 - iter * ((0.9 - 0.2) / max_iter);
    
    % 线性递减协同因子 my_c，从0.1递减到0
    my_c = 0.1 - iter * ((0.1 - 0) / (max_iter / 2));
    if my_c < 0
        my_c = 0;
    end
    
    % 根据当前迭代次数调整各项权重
    if iter < (3 * max_iter / 4)
        s = my_c;             % 分离权重
        a = my_c;             % 对齐权重
        c = my_c;             % 凝聚权重
        f = 2 * rand;         % 食物吸引权重
        e = my_c;             % 敌人驱散权重
    else
        s = my_c / iter;     % 分离权重
        a = my_c / iter;     % 对齐权重
        c = my_c / iter;     % 凝聚权重
        f = 2 * rand;        % 食物吸引权重
        e = my_c / iter;     % 敌人驱散权重
    end
    
    % %% 评估当前种群的适应度 %%
    for i = 1:N
        % 计算第 i 个蜻蜓的目标函数值
        Particles_F(i, :) = ObjectiveFunction(X(:, i)');
        
        % 更新食物（寻找支配当前食物的更优解）
        if dominates(Particles_F(i, :), Food_fitness)
            Food_fitness = Particles_F(i, :);
            Food_pos = X(:, i);
        end
        
        % 更新敌人（寻找被当前敌人支配的更劣解）
        if dominates(Enemy_fitness, Particles_F(i, :))
            if all(X(:, i) < ub') && all(X(:, i) > lb')
                Enemy_fitness = Particles_F(i, :);
                Enemy_pos = X(:, i);
            end
        end
    end
    
    % 更新存档（将当前种群中的非支配解加入存档）
    [Archive_X, Archive_F, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, X, Particles_F, Archive_member_no);
    
    % 如果存档超出最大容量，则处理存档（移除部分存档成员）
    if Archive_member_no > ArchiveMaxSize
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
    else
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    end
    
    % 重新计算存档成员的排名
    Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    
    % %% 选择食物和敌人 %%
    % 选择存档中在较少人群区域的成员作为食物，以提高覆盖度
    index = RouletteWheelSelection(1 ./ Archive_mem_ranks);
    if index == -1
        index = 1;
    end
    Food_fitness = Archive_F(index, :);
    Food_pos = Archive_X(index, :)';
       
    % 选择存档中在较多人群区域的成员作为敌人，以提高覆盖度
    index = RouletteWheelSelection(Archive_mem_ranks);
    if index == -1
        index = 1;
    end
    Enemy_fitness = Archive_F(index, :);
    Enemy_pos = Archive_X(index, :)';
    
    % %% 更新每个蜻蜓的位置和速度 %%
    for i = 1:N
        index = 0;          % 邻居索引计数
        neighbours_no = 0;  % 邻居数量计数
        
        clear Neighbours_V  % 清除邻居速度
        clear Neighbours_X  % 清除邻居位置
        
        % 找到第 i 个蜻蜓的邻居
        for j = 1:N
            Dist = distance(X(:, i), X(:, j));  % 计算第 i 和第 j 个蜻蜓之间的距离
            if (all(Dist <= r) && all(Dist ~= 0))  % 判断是否在邻域内且不为自身
                index = index + 1;
                neighbours_no = neighbours_no + 1;
                Neighbours_V(:, index) = DeltaX(:, j);  % 存储邻居的速度
                Neighbours_X(:, index) = X(:, j);       % 存储邻居的位置
            end
        end
        
        % %% 分离行为（Separation）%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.1) 分离向量 S
        S = zeros(dim, 1);  % 初始化分离向量
        if neighbours_no > 1
            for k = 1:neighbours_no
                S = S + (Neighbours_X(:, k) - X(:, i));  % 计算邻居位置与自身位置的差
            end
            S = -S;  % 分离向量为相反方向
        else
            S = zeros(dim, 1);  % 如果没有足够的邻居，则分离向量为零
        end
        
        % %% 对齐行为（Alignment）%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.2) 对齐向量 A
        if neighbours_no > 1
            A = (sum(Neighbours_V, 2)) / neighbours_no;  % 计算邻居速度的平均值
        else
            A = DeltaX(:, i);  % 如果没有足够的邻居，则对齐向量为自身速度
        end
        
        % %% 凝聚行为（Cohesion）%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.3) 凝聚向量 C
        if neighbours_no > 1
            C_temp = (sum(Neighbours_X, 2)) / neighbours_no;  % 计算邻居位置的平均值
        else
            C_temp = X(:, i);  % 如果没有足够的邻居，则凝聚向量为自身位置
        end
        
        C = C_temp - X(:, i);  % 凝聚向量指向邻居的中心
        
        % %% 吸引食物（Attraction to Food）%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.4) 食物吸引向量 F
        Dist2Attraction = distance(X(:, i), Food_pos(:, 1));  % 计算与食物的距离
        if all(Dist2Attraction <= r)
            F = Food_pos - X(:, i);  % 如果在邻域内，则计算吸引向量
            iter;  % 可选：记录当前迭代次数（无实际作用）
        else
            F = 0;  % 否则，食物吸引向量为零
        end
        
        % %% 避免敌人（Distraction from Enemy）%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.5) 敌人驱散向量 E
        Dist = distance(X(:, i), Enemy_pos(:, 1));  % 计算与敌人的距离
        if all(Dist <= r)
            E = Enemy_pos + X(:, i);  % 如果在邻域内，则计算驱散向量
        else
            E = zeros(dim, 1);  % 否则，驱散向量为零
        end
        
        % %% 边界处理 %%
        for tt = 1:dim
            if X(tt, i) > ub(tt)
                X(tt, i) = lb(tt);            % 超过上界则环绕到下界
                DeltaX(tt, i) = rand;         % 重置速度为随机值
            end
            if X(tt, i) < lb(tt)
                X(tt, i) = ub(tt);            % 低于下界则环绕到上界
                DeltaX(tt, i) = rand;         % 重置速度为随机值
            end
        end
        
        % %% 更新速度和位置 %%
        if any(Dist2Attraction > r)
            % 如果与食物的距离超过邻域半径，则执行随机游走
            if neighbours_no > 1
                for j = 1:dim
                    % 更新速度：结合惯性权重、对齐、凝聚和分离行为
                    DeltaX(j, i) = w * DeltaX(j, i) + rand * A(j, 1) + rand * C(j, 1) + rand * S(j, 1);
                    % 限制速度的最大值
                    if DeltaX(j, i) > V_max
                        DeltaX(j, i) = V_max;
                    end
                    if DeltaX(j, i) < -V_max
                        DeltaX(j, i) = -V_max;
                    end
                    % 更新位置
                    X(j, i) = X(j, i) + DeltaX(j, i);
                end
            else
                % 如果没有足够的邻居，则使用 Levy 飞行更新位置
                X(:, i) = X(:, i) + Levy(dim)' .* X(:, i);
                DeltaX(:, i) = 0;  % 重置速度
            end
        else
            % 否则，根据分离、对齐、凝聚、食物吸引和敌人驱散更新速度
            for j = 1:dim
                DeltaX(j, i) = s * S(j, 1) + a * A(j, 1) + c * C(j, 1) + f * F(j, 1) + e * E(j, 1) + w * DeltaX(j, i);
                % 限制速度的最大值
                if DeltaX(j, i) > V_max
                    DeltaX(j, i) = V_max;
                end
                if DeltaX(j, i) < -V_max
                    DeltaX(j, i) = -V_max;
                end
                % 更新位置
                X(j, i) = X(j, i) + DeltaX(j, i);
            end
        end
        
        % 再次进行边界处理，确保所有位置在定义域内
        Flag4ub = X(:, i) > ub';
        Flag4lb = X(:, i) < lb';
        X(:, i) = (X(:, i) .* (~(Flag4ub + Flag4lb))) + ub' .* Flag4ub + lb' .* Flag4lb;
    end
    
    % %% 显示当前迭代的存档信息 %%
    display(['At the iteration ', num2str(iter), ' there are ', num2str(Archive_member_no), ' non-dominated solutions in the archive']);
end
% %% 主循环结束 %%

% %% 结果可视化 %%
figure

% 绘制 ZDT1 的真实帕累托前沿
Draw_ZDT1();

hold on
% 绘制存档中的非支配解
if obj_no == 2
    plot(Archive_F(:, 1), Archive_F(:, 2), 'ko', 'MarkerSize', 8, 'markerfacecolor', 'k');
else
    plot3(Archive_F(:, 1), Archive_F(:, 2), Archive_F(:, 3), 'ko', 'MarkerSize', 8, 'markerfacecolor', 'k');
end
legend('True PF', 'Obtained PF');  % 添加图例
title('MODA');                     % 设置标题
