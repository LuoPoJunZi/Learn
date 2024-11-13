function o = dominates(x, y)
% 判断解 x 是否支配解 y
% 支配的定义是：
%   - x 的所有目标值都小于或等于 y 对应的目标值（即 x<=y）
%   - x 至少有一个目标值严格小于 y 对应的目标值（即 x<y）
% 输入:
%   x, y: 解向量，表示一组目标值
% 输出:
%   o: 布尔值，若 x 支配 y，则 o 为 true；否则为 false

% 判断解 x 是否支配解 y
o = all(x <= y) && any(x < y);
