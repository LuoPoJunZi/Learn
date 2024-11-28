% dominates.m - 判断一个解是否支配另一个解
%
% 输入:
%   x - 第一个解（向量）
%   y - 第二个解（向量）
%
% 输出:
%   o - 如果 x 支配 y，则为真；否则为假

function o = dominates(x, y)
    % 检查 x 是否在所有目标上都不劣于 y
    % 并且至少在一个目标上优于 y
    o = all(x <= y) && any(x < y);
end
