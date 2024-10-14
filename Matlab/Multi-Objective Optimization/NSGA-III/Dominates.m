function b = Dominates(x, y)
    % Dominates 函数用于检查一个解是否支配另一个解
    % 输入：
    %   x, y - 两个解（可以是包含目标值的结构体或目标值向量）
    % 输出：
    %   b - 布尔值，若 x 支配 y 则为 true，否则为 false

    % 如果 x 是结构体，则提取其目标值
    if isstruct(x)
        x = x.Cost;
    end

    % 如果 y 是结构体，则提取其目标值
    if isstruct(y)
        y = y.Cost;
    end

    % 检查 x 是否支配 y
    % x 支配 y 的条件是：
    % 1. x 的所有目标值都小于或等于 y 的目标值
    % 2. x 至少有一个目标值严格小于 y 的目标值
    b = all(x <= y) && any(x < y);
end
