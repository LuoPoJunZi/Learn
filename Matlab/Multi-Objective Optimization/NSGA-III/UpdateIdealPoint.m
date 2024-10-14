function zmin = UpdateIdealPoint(pop, prev_zmin)
    % 更新理想点（最优解点）
    % 输入：
    %   pop - 当前种群
    %   prev_zmin - 上一轮的理想点
    % 输出：
    %   zmin - 更新后的理想点

    % 如果没有提供 prev_zmin 或其为空，则初始化为无穷大
    if ~exist('prev_zmin', 'var') || isempty(prev_zmin)
        prev_zmin = inf(size(pop(1).Cost));  % 设置为与成本大小相同的无穷大
    end
    
    zmin = prev_zmin;  % 初始化理想点为前一轮的理想点
    for i = 1:numel(pop)
        % 在当前个体成本与当前理想点之间取最小值，更新理想点
        zmin = min(zmin, pop(i).Cost);
    end

end
