% Levy.m - 生成 Levy 飞行步长
%
% 输入:
%   d - 步长的维度
%
% 输出:
%   o - Levy 飞行的步长向量

function o = Levy(d)
    beta = 3/2; % Levy 指数
    
    % 计算 sigma，根据 Eq. (3.10)
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    
    % 生成随机数 u 和 v，符合标准正态分布
    u = randn(1, d) * sigma;
    v = randn(1, d);
    
    % 计算步长，根据 Eq. (3.10)
    step = u ./ abs(v).^(1 / beta);
    
    % 计算 Levy 飞行步长，根据 Eq. (3.9)
    o = 0.01 * step;
end
