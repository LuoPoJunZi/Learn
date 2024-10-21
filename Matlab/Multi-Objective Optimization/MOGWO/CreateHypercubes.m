% 函数 CreateHypercubes
% 用于为每个目标生成超立方体（或网格），这些网格用于非支配解的分类。

function G=CreateHypercubes(costs, ngrid, alpha)

    % 获取目标数目，即成本矩阵的行数
    nobj = size(costs, 1);
    
    % 创建一个空的网格结构体，用于存储每个目标的上下界
    empty_grid.Lower = [];    % 网格的下边界
    empty_grid.Upper = [];    % 网格的上边界
    G = repmat(empty_grid, nobj, 1);  % 为每个目标分配一个网格

    % 遍历每个目标
    for j = 1:nobj
        
        % 获取第 j 个目标的最小值和最大值
        min_cj = min(costs(j, :));
        max_cj = max(costs(j, :));
        
        % 计算扩展范围 dcj（扩展比例为 alpha）
        dcj = alpha * (max_cj - min_cj);
        
        % 扩展目标值的上下界
        min_cj = min_cj - dcj;
        max_cj = max_cj + dcj;
        
        % 使用 linspace 函数将目标值范围划分为 ngrid-1 个网格
        gx = linspace(min_cj, max_cj, ngrid - 1);
        
        % 设置第 j 个目标的网格上下界
        G(j).Lower = [-inf gx];   % 下边界数组，第一项为 -inf
        G(j).Upper = [gx inf];    % 上边界数组，最后一项为 inf
        
    end

end
