function f = replace_chromosome(intermediate_chromosome, M, D, NP)
%% 替换种群个体的函数
% 此函数基于个体的非支配等级（Rank）和拥挤距离（Crowding Distance）对种群进行替换。
% 在种群大小达到要求之前，每次添加一个前沿，直到添加的前沿导致种群大小超过要求为止。
% 在这种情况下，按照拥挤距离降序选择当前前沿的个体，直至填满种群。

% 输入参数：
% intermediate_chromosome - 中间种群矩阵，包含决策变量、目标值、Rank和拥挤距离
% M - 目标函数的数量
% D - 决策变量的数量
% NP - 所需种群大小（种群容量）

% 输出参数：
% f - 替换后的种群矩阵，大小为 NP 行

%% 初始化
[~, m] = size(intermediate_chromosome); % 获取种群矩阵的列数
f = zeros(NP, m); % 初始化输出种群矩阵

% 根据个体的 Rank（非支配等级）进行排序
sorted_chromosome = sortrows(intermediate_chromosome, M + D + 1);

% 找到当前种群中的最大 Rank 值
max_rank = max(intermediate_chromosome(:, M + D + 1));

% 逐个添加前沿，直到种群被填满
previous_index = 0;
for i = 1:max_rank
    % 找到当前 Rank 的最后一个个体的索引
    current_index = find(sorted_chromosome(:, M + D + 1) == i, 1, 'last');
    
    % 检查如果将当前 Rank 的所有个体添加到种群中是否会超出种群大小
    if current_index > NP
        % 如果会超出种群大小，则计算剩余的种群容量
        remaining = NP - previous_index;
        
        % 提取当前 Rank 的所有个体
        temp_pop = sorted_chromosome(previous_index + 1:current_index, :);
        
        % 根据拥挤距离降序对当前 Rank 的个体排序
        [~, temp_sort_index] = sort(temp_pop(:, M + D + 2), 'descend');
        
        % 按拥挤距离降序选择剩余的个体填充种群
        for j = 1:remaining
            f(previous_index + j, :) = temp_pop(temp_sort_index(j), :);
        end
        return; % 种群填满，退出函数
    elseif current_index < NP
        % 如果当前 Rank 的个体可以完全添加到种群中，则直接添加
        f(previous_index + 1:current_index, :) = ...
            sorted_chromosome(previous_index + 1:current_index, :);
    else
        % 如果当前 Rank 的个体正好填满种群，则直接添加并退出
        f(previous_index + 1:current_index, :) = ...
            sorted_chromosome(previous_index + 1:current_index, :);
        return;
    end
    % 更新前一个索引为当前索引，继续处理下一个 Rank
    previous_index = current_index;
end
