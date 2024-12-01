% BinaryTournamentSelection.m
% 二元锦标赛选择函数，用于从种群中选择适应度较好的个体

function p = BinaryTournamentSelection(pop, f)
    % 输入参数:
    % pop - 当前种群，包含多个个体
    % f - 个体的适应度向量
    
    % 输出:
    % p - 被选中的个体

    n = numel(pop);            % 获取种群中个体的数量
    
    I = randsample(n, 2);      % 随机抽取两个不同的个体索引
    
    i1 = I(1);                 % 第一个被抽中的个体索引
    i2 = I(2);                 % 第二个被抽中的个体索引
    
    % 比较两个个体的适应度，选择适应度较好的个体
    if f(i1) < f(i2)
        p = pop(i1);
    else
        p = pop(i2);
    end
end
