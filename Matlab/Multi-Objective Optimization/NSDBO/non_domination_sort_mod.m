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
