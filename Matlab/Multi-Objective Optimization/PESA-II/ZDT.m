function z = ZDT(x)
% ZDT 定义了ZDT多目标优化问题（具体版本未明确）
% 输入参数:
%   x - 决策变量向量
% 输出参数:
%   z - 目标值向量

    n = numel(x);  % 决策变量的数量
    
    % 第一个目标函数
    f1 = x(1);
    
    % 计算辅助函数g
    g = 1 + 9 / (n - 1) * sum(x(2:end));
    
    % 计算辅助函数h
    h = 1 - sqrt(f1 / g);
    
    % 第二个目标函数
    f2 = g * h;
    
    % 返回目标值向量
    z = [f1
         f2];
    
end