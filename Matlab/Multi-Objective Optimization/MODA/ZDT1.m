% ZDT1.m - ZDT1 测试函数
%
% ZDT1 是一个经典的多目标优化测试函数，具有两个目标和多个决策变量。
%
% 输入:
%   x - 决策变量向量（行向量）
%
% 输出:
%   o - 目标函数值向量（列向量），包含两个目标值

% 修改此文件以适应您的目标函数
function o = ZDT1(x)
    o = [0, 0]; % 初始化目标函数值向量
    
    dim = length(x); % 决策变量的维度
    % 计算辅助函数 g，根据 ZDT1 定义
    g = 1 + 9 * sum(x(2:dim)) / (dim - 1);
    
    % 计算第一个目标函数 f1
    o(1) = x(1);
    % 计算第二个目标函数 f2
    o(2) = g * (1 - sqrt(x(1) / g));
end
