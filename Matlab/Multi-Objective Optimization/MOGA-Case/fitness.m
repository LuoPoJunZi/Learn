% 适应度函数
function f = fitness(x)
% 输入参数:
% x - 1x9的参数向量，包含多个决策变量
% 输出:
% f - 向量输出，表示适应度值，包含多个目标的评估结果

    % 调用 my_model1 函数，计算目标值 y1 和 y2
    [y1, y2] = my_model1(x);
    
    % 将目标值组成输出向量 f
    f = [y1, y2];
end
