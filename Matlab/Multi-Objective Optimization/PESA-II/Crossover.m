function [y1, y2]=Crossover(x1, x2, params)
% Crossover 执行交叉操作，生成两个新的个体
% 输入参数:
%   x1, x2 - 父代个体的位置向量
%   params - 交叉操作的参数结构体，包含gamma, VarMin, VarMax
% 输出参数:
%   y1, y2 - 生成的子代个体的位置向量

    % 提取参数
    gamma = params.gamma;         % 控制交叉范围的参数
    VarMin = params.VarMin;       % 决策变量的下界
    VarMax = params.VarMax;       % 决策变量的上界
    
    % 生成交叉系数alpha，范围在[-gamma, 1+gamma]
    alpha = unifrnd(-gamma, 1 + gamma, size(x1));
    
    % 计算子代个体的位置
    y1 = alpha .* x1 + (1 - alpha) .* x2;
    y2 = alpha .* x2 + (1 - alpha) .* x1;
    
    % 确保子代的位置在允许范围内
    y1 = min(max(y1, VarMin), VarMax);
    y2 = min(max(y2, VarMin), VarMax);

end
