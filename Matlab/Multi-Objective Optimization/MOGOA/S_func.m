function o = S_func(r)
% S_func 计算公式 Eq. (3.3) 中的函数
% 输入:
%   r: 输入的距离值
% 输出:
%   o: 计算结果

F = 0.5;  % 常数 F
L = 1.5;  % 常数 L

% 计算函数值：o = F * exp(-r / L) - exp(-r)
o = F * exp(-r / L) - exp(-r);  % Eq. (3.3) in the paper
