%% 引用自 NSGA-II，版权所有
function f = initialize_variables(NP, M, D, LB, UB)
%% 初始化种群的函数
% 该函数用于初始化种群，每个个体在初始阶段具有：
%   * 一组决策变量（decision variables）
%   * 对应的目标函数值（objective function values）
%
% 输入参数：
% NP - 种群大小
% M  - 目标函数的数量
% D  - 决策变量的数量
% LB - 每个决策变量的下界（向量）
% UB - 每个决策变量的上界（向量）
%
% 输出：
% f - 包含所有个体决策变量和目标值的种群矩阵

%% 初始化变量
min = LB; % 决策变量的下界
max = UB; % 决策变量的上界
K = M + D; % 决策变量与目标函数值的总维度
f = zeros(NP, K); % 用于存储种群的矩阵，行对应个体，列对应变量和目标值

%% 初始化种群中的每个个体
for i = 1 : NP
    % 随机初始化每个个体的决策变量
    for j = 1 : D
        % 决策变量范围为 [LB(j), UB(j)]
        f(i, j) = min(j) + (max(j) - min(j)) * rand(1);
    end
    
    % 计算该个体的目标函数值
    % f(i, D + 1 : K) 存储目标函数值
    f(i, D + 1 : K) = evaluate_objective(f(i, 1:D));
end
