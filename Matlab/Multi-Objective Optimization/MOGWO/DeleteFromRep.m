% 函数 DeleteFromRep
% 该函数用于从外部存档 (rep) 中删除额外的解（粒子）。
% 参数：
%   - rep: 外部存档，即当前存储的非支配解的集合
%   - EXTRA: 需要删除的解的数量
%   - gamma: 控制删除策略的参数，默认为 1

function rep = DeleteFromRep(rep, EXTRA, gamma)

    % 如果没有传入 gamma 参数，则默认值为 1
    if nargin < 3
        gamma = 1;
    end

    % 循环执行 EXTRA 次，逐步从存档中删除解
    for k = 1:EXTRA
        % 获取已占据网格的索引及其成员数量
        [occ_cell_index, occ_cell_member_count] = GetOccupiedCells(rep);

        % 计算删除概率 p，成员数越多，删除的概率越大（由 gamma 控制）
        p = occ_cell_member_count .^ gamma;  % 提升成员数的次幂
        p = p / sum(p);  % 归一化为概率分布

        % 使用轮盘赌法选择一个网格进行删除操作
        selected_cell_index = occ_cell_index(RouletteWheelSelection(p));

        % 提取所有存档粒子的网格索引
        GridIndices = [rep.GridIndex];

        % 找到当前被选中的网格中所有的粒子
        selected_cell_members = find(GridIndices == selected_cell_index);

        % 获取该网格中的粒子数量
        n = numel(selected_cell_members);

        % 随机选择该网格中的一个粒子进行删除
        selected_memebr_index = randi([1 n]);

        % 获取该粒子在存档中的实际索引
        j = selected_cell_members(selected_memebr_index);
        
        % 删除该粒子，从存档中移除
        rep = [rep(1:j-1); rep(j+1:end)];
    end
    
end
