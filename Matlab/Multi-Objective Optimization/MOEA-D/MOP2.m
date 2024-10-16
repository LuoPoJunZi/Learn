% MOP2多目标优化函数
% 输入参数:
% x - 决策变量的向量

function z = MOP2(x)

    n = numel(x);  % 获取决策变量的数量
    
    % 计算两个目标函数值
    z = [
        1 - exp(-sum((x - 1/sqrt(n)).^2));  % 第一个目标函数
        1 - exp(-sum((x + 1/sqrt(n)).^2))   % 第二个目标函数
    ];
    
end
