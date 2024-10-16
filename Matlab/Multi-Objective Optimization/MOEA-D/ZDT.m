% ZDT多目标优化函数
% 输入参数:
% x - 决策变量的向量

function z = ZDT(x)

    n = numel(x);  % 获取决策变量的数量
    f1 = x(1);  % 第一个目标函数值为决策变量的第一个元素
    g = 1 + 9 / (n - 1) * sum(x(2:end));  % 计算g函数，涉及其他决策变量
    h = 1 - sqrt(f1 / g);  % 计算h函数
    f2 = g * h;  % 计算第二个目标函数值
    
    z = [f1;   % 返回目标函数值
         f2];
end
