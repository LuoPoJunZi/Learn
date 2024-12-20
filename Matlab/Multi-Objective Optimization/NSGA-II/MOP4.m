function z = MOP4(x)
    % MOP4：另一个测试问题的目标函数，用于多目标优化问题
    % 输入：
    %   x - 决策变量向量
    % 输出：
    %   z - 包含两个目标函数值的列向量

    a = 0.8;  % 参数a，用于目标函数2
    b = 3;    % 参数b，用于目标函数2
    
    % 目标函数1：基于相邻决策变量的平方和的非线性函数
    z1 = sum(-10 * exp(-0.2 * sqrt(x(1:end-1).^2 + x(2:end).^2)));
    % 对相邻的x(i)和x(i+1)进行组合，并通过指数函数构造一个最小化问题
    
    % 目标函数2：基于决策变量绝对值和正弦函数的组合
    z2 = sum(abs(x).^a + 5 * (sin(x)).^b);
    % 包含x的绝对值部分以及正弦函数的非线性项，目标同样是最小化
    
    % 返回两个目标函数值组成的列向量
    z = [z1 z2]';

end
