% initialization.m - 初始化搜索代理的初始种群
%
% 输入:
%   SearchAgents_no - 搜索代理的数量
%   dim - 决策变量的维度
%   ub - 决策变量的上界（可以是标量或向量）
%   lb - 决策变量的下界（可以是标量或向量）
%
% 输出:
%   X - 初始化后的种群矩阵，每列代表一个搜索代理的决策变量

function X = initialization(SearchAgents_no, dim, ub, lb)
    % 获取边界的数量（即决策变量的维度）
    Boundary_no = size(ub, 2); 
    
    % 如果所有决策变量的上下界相同且用户输入的是单个数字
    if Boundary_no == 1
        ub_new = ones(1, dim) * ub; % 创建一个与决策变量维度相同的上界向量
        lb_new = ones(1, dim) * lb; % 创建一个与决策变量维度相同的下界向量
    else
        ub_new = ub; % 使用用户提供的上界向量
        lb_new = lb; % 使用用户提供的下界向量
    end
    
    % 对于每个决策变量，生成在上下界之间的随机值
    for i = 1:dim
        ub_i = ub_new(i); % 第 i 个决策变量的上界
        lb_i = lb_new(i); % 第 i 个决策变量的下界
        X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i; % 生成随机值
    end
    
    X = X'; % 转置矩阵，使每列代表一个搜索代理
end
