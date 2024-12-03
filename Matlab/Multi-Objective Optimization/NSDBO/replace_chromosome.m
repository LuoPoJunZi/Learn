% replace_chromosome.m
% 替换染色体函数，用于根据排名和拥挤距离替换种群中的个体
% 参考自NSGA-II，版权所有。
% 输入:
%   intermediate_chromosome - 经过非支配排序和拥挤距离计算后的种群矩阵
%                            每一行包含决策变量、目标函数值、排名和拥挤距离
%   M                      - 目标函数的数量
%   D                      - 决策变量的数量
%   NP                     - 种群规模（Population Size）
% 输出:
%   f                      - 替换后的种群矩阵，大小为NP x (D + M + 2)
%                            每一行包含决策变量、目标函数值、排名和拥挤距离

function f = replace_chromosome(intermediate_chromosome, M, D, NP)

    %% 函数说明
    % 该函数根据个体的排名（Rank）和拥挤距离（Crowding Distance）替换种群中的个体。
    % 首先按排名对个体进行排序，依次添加每个前沿（Front）中的个体，
    % 直到达到种群规模NP。如果添加完整个前沿会超过NP，
    % 则根据拥挤距离从当前前沿中选择合适的个体填充到种群中。
    
    %% 按排名排序种群
    sorted_chromosome = sortrows(intermediate_chromosome, M + D + 1);
    % sorted_chromosome按第(M+D+1)列（排名）从小到大排序
    
    %% 获取最大排名
    max_rank = max(intermediate_chromosome(:, M + D + 1));
    % max_rank为当前种群中的最大排名值
    
    %% 初始化替换后的种群矩阵
    f = [];  % 初始化为空矩阵
    
    %% 逐前沿添加个体到种群
    previous_index = 0;  % 上一个前沿结束的索引
    for i = 1 : max_rank
        % 找到当前排名i的个体的最后一个索引
        current_index = find(sorted_chromosome(:, M + D + 1) == i, 1, 'last');
        
        if isempty(current_index)
            % 如果当前前沿没有个体，跳过
            continue;
        end
        
        % 判断添加当前前沿的个体是否会超过种群规模
        if (current_index > NP)
            % 计算剩余可添加的个体数量
            remaining = NP - previous_index;
            
            if remaining <= 0
                % 如果没有剩余空间，结束替换过程
                break;
            end
            
            % 提取当前前沿的所有个体
            temp_pop = sorted_chromosome(previous_index + 1 : current_index, :);
            
            % 按拥挤距离从高到低排序
            [~, temp_sort_index] = sort(temp_pop(:, M + D + 2), 'descend');
            % temp_sort_index是拥挤距离降序排列的索引
            
            % 从拥挤距离最高的个体开始添加，直到填满种群
            for j = 1 : remaining
                f = [f; temp_pop(temp_sort_index(j), :)];
                if j == remaining
                    break;
                end
            end
            
            % 填满种群后，结束替换过程
            return;
        elseif (current_index <= NP)
            % 如果添加当前前沿的所有个体不会超过种群规模，
            % 则将这些个体全部添加到替换后的种群中
            f = [f; sorted_chromosome(previous_index + 1 : current_index, :)];
        else
            % 如果添加当前前沿的所有个体会超过种群规模，
            % 则根据拥挤距离选择部分个体添加
            remaining = NP - previous_index;
            if remaining > 0
                temp_pop = sorted_chromosome(previous_index + 1 : current_index, :);
                [~, temp_sort_index] = sort(temp_pop(:, M + D + 2), 'descend');
                for j = 1 : remaining
                    f = [f; temp_pop(temp_sort_index(j), :)];
                    if j == remaining
                        break;
                    end
                end
            end
            return;
        end
        
        % 更新previous_index为当前前沿的最后一个索引
        previous_index = current_index;
    end
    
end
