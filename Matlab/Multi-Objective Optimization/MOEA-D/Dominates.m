% 判断个体x是否支配个体y的函数
% 输入参数:
% x - 被比较的个体
% y - 另一个被比较的个体

function b = Dominates(x, y)

    % 如果个体x是结构体，则提取代价值
    if isfield(x, 'Cost')
        x = x.Cost;  % 获取个体x的代价
    end

    % 如果个体y是结构体，则提取代价值
    if isfield(y, 'Cost')
        y = y.Cost;  % 获取个体y的代价
    end
    
    % 判断x是否支配y
    % x支配y当且仅当x在所有目标上都不大于y，并且至少在一个目标上严格小于y
    b = all(x <= y) && any(x < y);  

end
