function b=Dominates(x, y)
% Dominates 判断个体x是否支配个体y
% 支配条件: x在所有目标上都不劣于y，且至少在一个目标上优于y
% 输入参数:
%   x, y - 两个待比较的个体，包含Cost字段
% 输出参数:
%   b - 布尔值，若x支配y，则为true，否则为false

    % 如果个体包含Cost字段，则提取目标值
    if isfield(x, 'Cost')
        x = x.Cost;
    end
    
    if isfield(y, 'Cost')
        y = y.Cost;
    end

    % 检查是否在所有目标上x <= y且至少一个目标上x < y
    b = all(x <= y) && any(x < y);

end
