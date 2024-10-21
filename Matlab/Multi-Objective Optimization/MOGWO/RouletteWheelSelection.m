function i = RouletteWheelSelection(p)
    % 轮盘赌选择方法
    % 输入:
    %   p - 概率向量，表示每个个体被选择的概率
    % 输出:
    %   i - 被选择个体的索引

    r = rand;  % 生成一个在 [0, 1] 之间的随机数
    c = cumsum(p);  % 计算概率的累积和

    % 找到第一个累积和大于等于随机数的索引
    i = find(r <= c, 1, 'first');  
end
