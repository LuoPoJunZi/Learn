function y = Mutate(x, mu, sigma)
    % Mutate 函数用于执行突变操作，通过随机扰动来改变个体的基因
    % 输入：
    %   x - 决策变量向量（即个体）
    %   mu - 突变率，表示将有多少比例的基因发生突变
    %   sigma - 突变强度，表示突变时的标准差（变化量）
    % 输出：
    %   y - 突变后的个体

    % 获取决策变量的数量
    nVar = numel(x);
    
    % 根据突变率 mu 计算需要突变的基因数量 nMu
    nMu = ceil(mu * nVar);

    % 随机选择 nMu 个基因的位置 j 进行突变
    j = randsample(nVar, nMu);
    
    % 将原始个体 x 赋值给 y，确保 y 基本结构不变
    y = x;
    
    % 在随机选定的基因位置上，进行高斯分布的随机扰动
    % 使用 sigma 控制突变强度，randn 生成正态分布的随机数
    y(j) = x(j) + sigma * randn(size(j));

end
