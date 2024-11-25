%% 引用自 NSGA-II，版权所有
function f = non_domination_sort_mod(x, M, D)
%% 函数功能：非支配排序（Non-Domination Sorting）
% 此函数根据非支配关系对种群进行排序。
% 每个前沿中的个体被分配一个等级（Rank），第一前沿为1，第二前沿为2，依次类推。
% 在分配等级后，计算每个前沿的拥挤距离（Crowding Distance）。

% 输入参数：
% x - 当前种群矩阵（包含决策变量和目标值）
% M - 目标函数的数量
% D - 决策变量的数量

% 输出参数：
% f - 排序后的种群矩阵，包括非支配等级和拥挤距离

%% 初始化
[N, ~] = size(x); % N 为种群大小
front = 1; % 当前前沿编号
F(front).f = []; % 用于存储每个前沿的个体索引
individual = []; % 存储每个个体的支配信息

%% 非支配排序
for i = 1:N
    % 初始化每个个体的信息
    individual(i).n = 0; % 被其他个体支配的数量
    individual(i).p = []; % 当前个体支配的其他个体索引列表
    for j = 1:N
        dom_less = 0;
        dom_equal = 0;
        dom_more = 0;
        % 比较两个个体的每个目标函数值
        for k = 1:M
            if x(i, D+k) < x(j, D+k)
                dom_less = dom_less + 1;
            elseif x(i, D+k) == x(j, D+k)
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        % 更新支配信息
        if dom_less == 0 && dom_equal ~= M
            individual(i).n = individual(i).n + 1; % 被支配
        elseif dom_more == 0 && dom_equal ~= M
            individual(i).p = [individual(i).p j]; % 支配其他个体
        end
    end
    % 如果当前个体未被支配，加入第一前沿
    if individual(i).n == 0
        x(i, M+D+1) = 1; % Rank 为 1
        F(front).f = [F(front).f i];
    end
end

%% 确定后续前沿
while ~isempty(F(front).f)
    Q = []; % 用于存储下一前沿的个体
    for i = 1:length(F(front).f)
        % 更新当前前沿中个体支配的其他个体信息
        if ~isempty(individual(F(front).f(i)).p)
            for j = 1:length(individual(F(front).f(i)).p)
                individual(individual(F(front).f(i)).p(j)).n = ...
                    individual(individual(F(front).f(i)).p(j)).n - 1;
                if individual(individual(F(front).f(i)).p(j)).n == 0
                    x(individual(F(front).f(i)).p(j), M+D+1) = front + 1; % 设置 Rank
                    Q = [Q individual(F(front).f(i)).p(j)];
                end
            end
        end
    end
    front = front + 1; % 更新前沿编号
    F(front).f = Q; % 设置下一前沿的个体
end

% 按前沿排序
sorted_based_on_front = sortrows(x, M+D+1);

%% 计算拥挤距离
current_index = 0;
for front = 1:(length(F)-1)
    y = [];
    previous_index = current_index + 1;
    for i = 1:length(F(front).f)
        y(i, :) = sorted_based_on_front(current_index + i, :);
    end
    current_index = current_index + i;
    for i = 1:M
        [sorted_based_on_objective, index_of_objectives] = sortrows(y, D+i);
        f_max = sorted_based_on_objective(end, D+i);
        f_min = sorted_based_on_objective(1, D+i);
        y(index_of_objectives(end), M+D+1+i) = Inf; % 边界个体的距离设为无穷大
        y(index_of_objectives(1), M+D+1+i) = Inf;
        for j = 2:(length(index_of_objectives)-1)
            next_obj = sorted_based_on_objective(j+1, D+i);
            previous_obj = sorted_based_on_objective(j-1, D+i);
            if f_max - f_min == 0
                y(index_of_objectives(j), M+D+1+i) = Inf; % 避免除零错误
            else
                y(index_of_objectives(j), M+D+1+i) = ...
                    (next_obj - previous_obj) / (f_max - f_min);
            end
        end
    end
    % 计算总拥挤距离
    distance = zeros(length(F(front).f), 1);
    for i = 1:M
        distance = distance + y(:, M+D+1+i);
    end
    y(:, M+D+2) = distance;
    y = y(:, 1:M+D+2); % 截取有效列
    z(previous_index:current_index, :) = y;
end

% 返回排序后的种群
f = z;
