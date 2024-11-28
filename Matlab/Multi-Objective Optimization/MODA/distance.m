% distance.m - 计算两个向量之间的距离
%
% 输入:
%   a - 第一个向量
%   b - 第二个向量
%
% 输出:
%   o - 向量 a 和向量 b 之间每个维度的欧氏距离

function o = distance(a, b)
    % 遍历向量 a 的每一行
    for i = 1:size(a, 1)
        % 计算 a 和 b 在第 i 维度上的欧氏距离
        o(1, i) = sqrt((a(i) - b(i))^2);
    end
end
