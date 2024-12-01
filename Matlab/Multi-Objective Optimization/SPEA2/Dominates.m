% Dominates.m
% 支配关系判断函数，用于判断一个解是否支配另一个解

function b = Dominates(x, y)
    % 输入参数:
    % x, y - 两个待比较的个体，可以是结构体或向量
    
    % 输出:
    % b - 布尔值，表示x是否支配y

    % 如果输入是结构体且包含Cost字段，提取Cost向量
    if isstruct(x) && isfield(x, 'Cost')
        x = x.Cost;
    end

    if isstruct(y) && isfield(y, 'Cost')
        y = y.Cost;
    end

    % 判断x是否在所有目标上不劣于y，并且至少在一个目标上优于y
    b = all(x <= y) && any(x < y);
end
