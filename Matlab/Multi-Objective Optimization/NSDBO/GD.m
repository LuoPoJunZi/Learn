% GD.m
% 计算种群目标解与真实Pareto前沿之间的广义差距（Generational Distance）
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
%   PF     - 真实Pareto前沿的目标函数值矩阵
% 输出:
%   Score  - 广义差距值

function Score = GD(PopObj, PF)
    % 计算种群中每个个体到真实Pareto前沿的最小欧几里得距离
    Distance = min(pdist2(PopObj, PF), [], 2);
    
    % 计算所有最小距离的欧几里得范数，并除以种群大小以得到广义差距
    Score    = norm(Distance) / length(Distance);
end
