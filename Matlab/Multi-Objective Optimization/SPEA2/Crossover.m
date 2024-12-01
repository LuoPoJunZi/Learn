% Crossover.m
% 交叉操作函数，用于生成新的后代个体

function [y1, y2] = Crossover(x1, x2, params)
    % 输入参数:
    % x1, x2 - 父代个体的位置向量
    % params - 包含交叉参数的结构体，包含gamma, VarMin, VarMax
    
    % 输出:
    % y1, y2 - 生成的两个子代个体的位置向量

    gamma = params.gamma;       % 交叉参数gamma，用于控制交叉范围
    VarMin = params.VarMin;     % 变量的最小值
    VarMax = params.VarMax;     % 变量的最大值
    
    % 生成与父代个体相同大小的随机alpha值，范围在[-gamma, 1+gamma]
    alpha = unifrnd(-gamma, 1 + gamma, size(x1));
    
    % 生成子代个体的位置向量
    y1 = alpha .* x1 + (1 - alpha) .* x2;
    y2 = alpha .* x2 + (1 - alpha) .* x1;
    
    % 确保子代个体的位置在允许的范围内
    y1 = min(max(y1, VarMin), VarMax);
    y2 = min(max(y2, VarMin), VarMax);
end
