function [pop, grid]=FindPositionInGrid(pop, grid)
% FindPositionInGrid 将种群中的个体分配到对应的网格中
% 输入参数:
%   pop - 当前种群，包含Cost字段
%   grid - 网格结构数组，包含LB和UB
% 输出参数:
%   pop - 更新后的种群，包含GridIndex
%   grid - 更新后的网格，包含成员信息

    % 提取所有网格的下界和上界
    LB = [grid.LB];
    UB = [grid.UB];
    
    % 初始化每个网格中的个体数量和成员列表
    for k = 1:numel(grid)
        grid(k).N = 0;
        grid(k).Members = [];
    end
    
    % 遍历种群中的每个个体，找到其所在的网格
    for i = 1:numel(pop)
        % 找到个体的网格索引
        k = FindGridIndex(pop(i).Cost, LB, UB);
        pop(i).GridIndex = k;  % 记录个体的网格索引
        
        % 更新网格中的个体数量和成员列表
        grid(k).N = grid(k).N + 1;
        grid(k).Members = [grid(k).Members, i];
    end

end

function k=FindGridIndex(z, LB, UB)
% FindGridIndex 根据个体的目标值z找到其所在的网格索引
% 输入参数:
%   z - 个体的目标值向量
%   LB, UB - 所有网格的下界和上界
% 输出参数:
%   k - 网格的线性索引

    nObj = numel(z);        % 目标数量
    
    nGrid = size(LB, 2);    % 每个目标的网格数量
    f = true(1, nGrid);     % 初始化筛选条件为全部真
    
    % 对每个目标，判断z是否在对应网格的范围内
    for j = 1:nObj
        f = f & (z(j) >= LB(j, :)) & (z(j) < UB(j, :));
    end
    
    % 找到满足所有目标条件的网格索引
    k = find(f);

end
