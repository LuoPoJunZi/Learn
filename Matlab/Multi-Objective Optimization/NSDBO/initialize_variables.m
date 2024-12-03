% initialize_variables.m
% 初始化种群的决策变量和目标函数值
% 参考自NSGA-II，版权所有。
% 输入:
%   NP                - 种群规模（Population size）
%   M                 - 目标函数数量（Number of objective functions）
%   D                 - 决策变量数量（Number of decision variables）
%   LB                - 决策变量下界（Lower bounds）
%   UB                - 决策变量上界（Upper bounds）
%   evaluate_objective - 目标函数评估函数句柄
% 输出:
%   f                 - 初始化后的种群矩阵，包含决策变量和目标函数值

function f = initialize_variables(NP, M, D, LB, UB, evaluate_objective)
    
    % 将下界和上界赋值给min和max
    min = LB;
    max = UB;
    
    % K 是数组元素的总数。为了便于计算，决策变量和目标函数被连接成一个单一的数组。
    % 交叉和变异仅使用决策变量，而选择仅使用目标函数。
    K = M + D;
    
    % 初始化种群矩阵，大小为NP x K，初始值为零
    f = zeros(NP, K);
    
    %% 初始化种群中的每个个体
    % 对于每个染色体执行以下操作（N是种群规模）
    for i = 1 : NP
        % 根据决策变量的最小和最大可能值初始化决策变量。
        % 对于每个决策变量，随机生成一个在[min(j), max(j)]区间内的值
        for j = 1 : D
            f(i, j) = min(j) + (max(j) - min(j)) * rand(1);
        end % 结束j循环
        
        % 为了便于计算和数据处理，染色体还在末尾连接了目标函数值。
        % 从D+1到K的元素存储目标函数值。
        % evaluate_objective函数一次评估一个个体的目标函数值，
        % 仅传递决策变量，并返回目标函数值。这些值存储在个体的末尾。
        f(i, D + 1 : K) = evaluate_objective(f(i, 1:D));
    end % 结束i循环
end
