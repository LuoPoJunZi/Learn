% Spacing.m
% 计算种群的间距（Spacing）指标
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
% 输出:
%   Score  - 种群的间距值，反映解集的分布均匀性

function Score = Spacing(PopObj)

    % 计算种群中每对个体之间的曼哈顿距离（Cityblock Distance）
    Distance = pdist2(PopObj, PopObj, 'cityblock');
    
    % 将距离矩阵的对角线元素（即个体与自身的距离）设为无穷大，避免在后续计算中被选中
    Distance(logical(eye(size(Distance, 1)))) = inf;
    
    % 对于每个个体，找到其与其他所有个体的最小距离
    minDistances = min(Distance, [], 2);
    
    % 计算所有个体的最小距离的标准差，作为间距指标
    % 标准差越小，说明解集分布越均匀
    Score = std(minDistances);
end
