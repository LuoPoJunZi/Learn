function z = MOP2(x)
% MOP2 定义了一个多目标优化问题（示例）
% 输入参数:
%   x - 决策变量向量
% 输出参数:
%   z - 目标值向量

    n = numel(x);  % 决策变量的数量
    
    % 计算第一个目标
    z1 = 1 - exp(-sum((x - 1 / sqrt(n)).^2));
    
    % 计算第二个目标
    z2 = 1 - exp(-sum((x + 1 / sqrt(n)).^2));
    
    % 返回目标值向量
    z = [z1
         z2];
    
end
