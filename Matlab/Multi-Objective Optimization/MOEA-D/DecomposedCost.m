% 计算分解代价的函数
% 输入参数:
% individual - 当前个体，可以是代价向量或结构体
% z - 理想点（目标函数的真实值）
% lambda - 权重向量，用于分解计算

function g = DecomposedCost(individual, z, lambda)

    % 检查个体是否包含代价值
    if isfield(individual, 'Cost')
        fx = individual.Cost;  % 从结构体中提取代价值
    else
        fx = individual;  % 如果不是结构体，直接使用个体作为代价
    end
    
    % 计算分解代价，使用加权绝对差的最大值
    g = max(lambda .* abs(fx - z));  % 计算当前个体与理想点的代价

end
