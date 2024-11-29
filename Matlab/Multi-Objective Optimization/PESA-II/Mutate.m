function y = Mutate(x, params)
% Mutate 对个体x执行变异操作
% 输入参数:
%   x - 原始个体的位置向量
%   params - 变异操作的参数结构体，包含h, VarMin, VarMax
% 输出参数:
%   y - 变异后的个体的位置向量

    % 提取参数
    h = params.h;             % 变异步长因子
    VarMin = params.VarMin;   % 决策变量的下界
    VarMax = params.VarMax;   % 决策变量的上界
    
    % 计算变异的标准差
    sigma = h * (VarMax - VarMin);
    
    % 执行高斯变异
    y = x + sigma .* randn(size(x));
    
    % 或者执行均匀变异（注释掉）
    % y = x + sigma .* unifrnd(-1, 1, size(x));
    
    % 确保变异后的个体在允许范围内
    y = min(max(y, VarMin), VarMax);

end
