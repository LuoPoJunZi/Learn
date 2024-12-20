% MOP2.m
% 多目标优化问题2的目标函数

function z = MOP2(x)
    % 输入参数:
    % x - 决策变量向量
    
    % 输出:
    % z - 目标函数值向量

    n = numel(x);  % 决策变量的数量
    
    % 计算第一个目标函数值
    z1 = 1 - exp(-sum((x - 1 / sqrt(n)).^2));
    
    % 计算第二个目标函数值
    z2 = 1 - exp(-sum((x + 1 / sqrt(n)).^2));
    
    % 返回目标函数值向量
    z = [z1
         z2];
end
