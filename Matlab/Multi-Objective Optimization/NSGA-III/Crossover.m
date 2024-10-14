function [y1, y2] = Crossover(x1, x2)
    % Crossover 函数执行模拟二进制交叉操作（SBX），生成两个子代个体
    % 输入：
    %   x1, x2 - 父代个体的决策变量向量
    % 输出：
    %   y1, y2 - 子代个体的决策变量向量

    alpha = rand(size(x1));  % 生成与决策变量尺寸相同的随机权重系数 alpha
    
    % 计算第一个子代 y1，公式：y1 = alpha * x1 + (1 - alpha) * x2
    y1 = alpha .* x1 + (1 - alpha) .* x2;
    
    % 计算第二个子代 y2，公式：y2 = alpha * x2 + (1 - alpha) * x1
    y2 = alpha .* x2 + (1 - alpha) .* x1;
    
end
