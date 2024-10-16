% 交叉操作函数
% 输入参数:
% x1 - 第一个父代个体
% x2 - 第二个父代个体
% params - 参数结构体，包含交叉参数和变量范围

function y = Crossover(x1, x2, params)

    % 从参数中提取交叉相关参数
    gamma = params.gamma;       % 交叉幅度参数
    VarMin = params.VarMin;     % 变量的最小值
    VarMax = params.VarMax;     % 变量的最大值
    
    % 生成交叉因子，范围在[-gamma, 1 + gamma]之间
    alpha = unifrnd(-gamma, 1 + gamma, size(x1));
    
    % 进行线性组合生成子代个体
    y = alpha .* x1 + (1 - alpha) .* x2;

    % 确保生成的个体在变量范围内
    y = min(max(y, VarMin), VarMax);
    
end
