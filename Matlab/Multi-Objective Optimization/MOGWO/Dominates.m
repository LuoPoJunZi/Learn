% 函数 Dominates
% 该函数用于判断解 x 是否支配解 y。支配是多目标优化中的核心概念，
% 一个解 x 支配解 y 当且仅当：
% 1. 解 x 在所有目标上不劣于解 y；
% 2. 解 x 在至少一个目标上优于解 y。
% 参数：
%   - x: 解 x，包含目标值 (Cost)，可以是结构体也可以是向量
%   - y: 解 y，包含目标值 (Cost)，可以是结构体也可以是向量
% 返回：
%   - dom: 布尔值，若 x 支配 y，返回 true；否则返回 false

function dom = Dominates(x, y)

    % 如果 x 是结构体，则提取其目标值 (Cost)
    if isstruct(x)
        x = x.Cost;
    end

    % 如果 y 是结构体，则提取其目标值 (Cost)
    if isstruct(y)
        y = y.Cost;
    end
    
    % 判断支配条件：
    % 1. all(x <= y): 确保 x 在所有目标上不劣于 y
    % 2. any(x < y): 确保 x 在至少一个目标上优于 y
    dom = all(x <= y) && any(x < y);

end
