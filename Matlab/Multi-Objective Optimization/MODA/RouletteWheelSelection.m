% RouletteWheelSelection.m - 轮盘赌选择
%
% 该函数根据给定的权重向量进行轮盘赌选择，返回被选中的索引。
%
% 输入:
%   weights - 权重向量，每个元素代表对应解被选中的权重
%
% 输出:
%   o - 被选中的解的索引。如果权重为空或总和为零，则返回 -1

function o = RouletteWheelSelection(weights)
    % 计算权重的累积和
    accumulation = cumsum(weights);
    
    % 如果总权重为零，则无法进行选择
    if accumulation(end) == 0
        o = -1;
        return;
    end
    
    % 生成一个介于 0 和累积和之间的随机数
    p = rand() * accumulation(end);
    
    chosen_index = -1; % 初始化被选中的索引为 -1
    
    % 遍历累积和，找到第一个大于随机数 p 的索引
    for index = 1:length(accumulation)
        if (accumulation(index) > p)
            chosen_index = index;
            break;
        end
    end
    
    o = chosen_index; % 返回被选中的索引
end
