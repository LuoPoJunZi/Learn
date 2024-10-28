function o = dominates(x, y)
% dominates 函数用于判断解 x 是否支配解 y
% 输入：
%   x - 解 x 的目标函数值向量
%   y - 解 y 的目标函数值向量
% 输出：
%   o - 如果解 x 支配解 y，则返回 true；否则返回 false

% 支配条件：x 的每一个目标值都小于等于 y，且至少有一个目标值严格小于 y
o = all(x <= y) && any(x < y);

end
