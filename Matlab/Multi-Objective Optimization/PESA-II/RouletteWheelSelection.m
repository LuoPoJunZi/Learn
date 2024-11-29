function i = RouletteWheelSelection(p)
% RouletteWheelSelection 使用轮盘赌选择法选择个体
% 输入参数:
%   p - 每个个体的选择概率向量
% 输出参数:
%   i - 被选择的个体的索引

    r = rand * sum(p);         % 生成一个0到总概率之间的随机数
    c = cumsum(p);             % 计算累积概率
    i = find(r <= c, 1, 'first');  % 找到第一个累积概率大于或等于r的位置

end
