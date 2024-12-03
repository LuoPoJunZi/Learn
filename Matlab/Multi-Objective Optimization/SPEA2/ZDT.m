% ZDT.m
% ZDT 测试函数，用于多目标优化问题

function z = ZDT(x)
    % 输入参数:
    % x - 决策变量向量
    
    % 输出:
    % z - 目标函数值向量

    n = numel(x);      % 决策变量的数量

    f1 = x(1);         % 第一个目标函数值，通常与第一个决策变量相关
    
    % 计算辅助函数g，与后续的目标函数值相关
    g = 1 + 9 / (n - 1) * sum(x(2:end));
    
    % 计算辅助函数h，决定了两个目标函数之间的关系
    h = 1 - sqrt(f1 / g);
    
    f2 = g * h;        % 第二个目标函数值
    
    % 返回目标函数值向量
    z = [f1
         f2];
end