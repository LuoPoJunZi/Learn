function [sol, val] = gabpEval(sol, ~)
% gabpEval - 评估函数，用于遗传算法中计算个体的适应度
%
% 输入参数:
%   sol - 当前个体的参数向量
%   ~   - 占位符参数（未使用）
%
% 输出参数:
%   sol - 当前个体的参数向量（保持不变）
%   val - 当前个体的适应度值

    %% 解码适应度值
    val = gadecod(sol); % 使用 gadecod 函数计算适应度值
end
