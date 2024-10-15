function b = Dominates(x, y)
    % 判断解x是否支配解y，基于Pareto支配的概念
    % 输入：
    %   x - 解向量或包含目标函数值的结构体
    %   y - 解向量或包含目标函数值的结构体
    % 输出：
    %   b - 布尔值，若x支配y则返回true，否则返回false

    if isstruct(x)
        x = x.Cost;  % 如果x是结构体，提取其目标函数值
    end

    if isstruct(y)
        y = y.Cost;  % 如果y是结构体，提取其目标函数值
    end

    % 支配条件：
    % 1. x的所有目标函数值小于等于y的目标函数值（all(x<=y)）
    % 2. 至少有一个目标函数值x严格小于y（any(x<y)）
    b = all(x <= y) && any(x < y);  % 如果两个条件都满足，则x支配y
    
end
