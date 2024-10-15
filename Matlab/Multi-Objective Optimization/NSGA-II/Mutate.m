function y = Mutate(x, mu, sigma)
    % 执行突变操作，用于遗传算法中的个体变异
    % 输入：
    %   x - 原始个体的解向量
    %   mu - 突变概率，表示突变的决策变量比例
    %   sigma - 突变强度，控制突变的幅度（标准差）
    % 输出：
    %   y - 突变后的个体解向量

    nVar = numel(x);  % 获取决策变量的数量
    
    nMu = ceil(mu * nVar);  % 根据突变概率mu计算突变变量的数量
    
    % 随机选择nMu个要突变的决策变量
    j = randsample(nVar, nMu);  
    
    % 如果sigma是向量，确保只针对选择的决策变量应用对应的sigma值
    if numel(sigma) > 1
        sigma = sigma(j);
    end
    
    y = x;  % 复制原始个体
    
    % 对选择的决策变量应用高斯噪声进行突变
    y(j) = x(j) + sigma .* randn(size(j));  
    
end
