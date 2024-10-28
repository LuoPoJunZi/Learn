function [Best_universe_Inflation_rate, Best_universe, Archive_F] = MOMVO(Max_time, N, ArchiveMaxSize)
% MOMVO 函数实现了多目标多元宇宙优化算法 (MOMVO)
% 输入：
%   Max_time - 最大迭代次数
%   N - 搜索代理数量（解的数量）
%   ArchiveMaxSize - 归档的最大容量
% 输出：
%   Best_universe_Inflation_rate - 最优解的膨胀率（即最优目标函数值）
%   Best_universe - 最优解的位置
%   Archive_F - 最终非支配解的目标函数值归档

fobj = @Obj_function; % 目标函数句柄
dim = 4;              % 决策变量维数
lb = [0.125 0.1  0.1 0.125]; % 下边界
ub = [5.0 10.0 10.0 5.0];    % 上边界
obj_no = 2;           % 目标函数数量

% 如果上下界为单个数值，则扩展为维度相同的向量
if size(ub, 2) == 1
    ub = ones(1, dim) * ub;
    lb = ones(1, dim) * lb;
end

% 初始化最优解的位置和膨胀率（对多目标问题用 inf 表示极差初值）
Best_universe = zeros(1, dim);
Best_universe_Inflation_rate = inf * ones(1, obj_no);

% 初始化归档
Archive_X = zeros(ArchiveMaxSize, dim);
Archive_F = ones(ArchiveMaxSize, obj_no) * inf;
Archive_member_no = 0;

% 初始化 Wormhole Existence Probability (WEP) 的最大和最小值
WEP_Max = 1;
WEP_Min = 0.2;

% 初始化搜索代理的位置
Universes = initialization(N, dim, ub, lb);
Time = 1; % 初始化迭代计数器

% 主循环
while Time < Max_time + 1

    % 根据 Eq. (3.3) 计算当前迭代的 Wormhole Existence Probability
    WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_time);

    % 计算 Traveling Distance Rate (TDR) (Eq. (3.4))
    TDR = 1 - (Time^(1/6) / (Max_time)^(1/6));

    % 计算每个解的膨胀率（即目标函数值）
    for i = 1:size(Universes, 1)

        % 边界检查，将越界解调整回搜索空间
        Flag4ub = Universes(i, :) > ub;
        Flag4lb = Universes(i, :) < lb;
        Universes(i, :) = (Universes(i, :) .* ~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb;

        % 计算当前解的膨胀率
        Inflation_rates(i, :) = fobj(Universes(i, :));

        % 精英主义策略：更新最优解
        if dominates(Inflation_rates(i, :), Best_universe_Inflation_rate)
            Best_universe_Inflation_rate = Inflation_rates(i, :);
            Best_universe = Universes(i, :);
        end
    end

    % 对膨胀率排序并重新排列解
    [sorted_Inflation_rates, sorted_indexes] = sort(Inflation_rates);
    for newindex = 1:N
        Sorted_universes(newindex, :) = Universes(sorted_indexes(newindex), :);
    end

    % 归一化膨胀率，用于选择解
    normalized_sorted_Inflation_rates = normr(sorted_Inflation_rates);
    Universes(1, :) = Sorted_universes(1, :); % 保留当前的精英解

    % 更新归档
    [Archive_X, Archive_F, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, Universes, Inflation_rates, Archive_member_no);
    if Archive_member_no > ArchiveMaxSize
        % 调用 HandleFullArchive 处理归档超出容量情况
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
    else
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    end

    % 更新最优解
    index = RouletteWheelSelection(1 ./ Archive_mem_ranks);
    if index == -1
        index = 1;
    end
    Best_universe_Inflation_rate = Archive_F(index, :);
    Best_universe = Archive_X(index, :);

    % 更新每个解的位置
    for i = 2:size(Universes, 1) % 从第 2 个解开始（第 1 个为精英解）
        Back_hole_index = i;
        for j = 1:size(Universes, 2)
            r1 = rand();
            if r1 < normalized_sorted_Inflation_rates(i)
                % Eq. (3.1)：选择白洞个体
                White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates);
                if White_hole_index == -1
                    White_hole_index = 1;
                end
                Universes(Back_hole_index, j) = Sorted_universes(White_hole_index, j);
            end

            % Eq. (3.2)：根据边界更新解
            r2 = rand();
            if r2 < WEP
                r3 = rand();
                if r3 < 0.5
                    Universes(i, j) = Best_universe(1, j) + TDR * ((ub(j) - lb(j)) * rand + lb(j));
                else
                    Universes(i, j) = Best_universe(1, j) - TDR * ((ub(j) - lb(j)) * rand + lb(j));
                end
            end
        end
    end
    disp(['在第 ', num2str(Time), ' 次迭代中，归档中有 ', num2str(Archive_member_no), ' 个非支配解']);
    Time = Time + 1;
end
