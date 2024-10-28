function o = RouletteWheelSelection(weights)
% 基于轮盘赌选择法，从权重中随机选择一个索引
% 输入：
%   weights - 权重数组，表示每个选项被选择的相对概率
% 输出：
%   o - 被选择的索引，如果选择失败则返回 -1

% 计算权重的累积和
accumulation = cumsum(weights);

% 随机生成一个数，范围为 [0, accumulation 的总和]
p = rand() * accumulation(end);

% 初始化选择索引为 -1（未选择）
chosen_index = -1;

% 遍历累积和数组，找到第一个大于 p 的索引
for index = 1 : length(accumulation)
    if accumulation(index) > p
        chosen_index = index; % 设置被选择的索引
        break; % 一旦找到就退出循环
    end
end

% 输出选择的索引
o = chosen_index;
end
