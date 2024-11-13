function d = distance(a,b)
% 计算两个点 a 和 b 之间的欧氏距离
% 输入:
%   a, b: 2维坐标点，格式为 [x, y]
% 输出:
%   d: a 和 b 之间的欧氏距离

% 计算欧氏距离公式: d = sqrt((x1 - x2)^2 + (y1 - y2)^2)
d = sqrt((a(1) - b(1))^2 + (a(2) - b(2))^2);
