% Mutate.m
% 变异操作函数，用于对个体的位置向量进行变异

function y = Mutate(x, params)
    % 输入参数:
    % x - 待变异的个体的位置向量
    % params - 包含变异参数的结构体，包含h, VarMin, VarMax
    
    % 输出:
    % y - 变异后的个体的位置向量

    h = params.h;               % 变异参数h，用于控制变异幅度
    VarMin = params.VarMin;     % 变量的最小值
    VarMax = params.VarMax;     % 变量的最大值

    sigma = h * (VarMax - VarMin);    % 计算标准差，用于正态分布变异
    
    % 对位置向量进行正态分布变异
    y = x + sigma * randn(size(x));
    
    % 另一种变异方式：均匀分布变异
    % y = x + sigma * unifrnd(-1, 1, size(x));
    
    % 确保变异后的个体位置在允许范围内
    y = min(max(y, VarMin), VarMax);
end
