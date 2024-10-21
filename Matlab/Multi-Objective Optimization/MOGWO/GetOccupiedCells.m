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
