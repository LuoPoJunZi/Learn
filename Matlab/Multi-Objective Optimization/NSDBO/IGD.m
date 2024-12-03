% IGD.m
% 计算种群的反向广义差距（Inverted Generational Distance, IGD）指标
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
%   PF     - 真实Pareto前沿的目标函数值矩阵
% 输出:
%   Score  - 反向广义差距值

function Score = IGD(PopObj, PF)
    % 计算每个真实Pareto前沿点到种群中最近个体的欧几里得距离
    Distance = min(pdist2(PF, PopObj), [], 2);
    
    % 计算所有距离的平均值，得到反向广义差距
    Score    = mean(Distance);
end
