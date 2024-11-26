% -------------------------------------------------------------
% 文件名: initialization.m
% 功能: 初始化种群的位置（决策变量），在给定的上下界范围内随机生成
% 输入:
%       SearchAgents_no - 搜索代理（个体）的数量
%       dim             - 决策变量的维度
%       ub              - 决策变量的上界（标量或向量）
%       lb              - 决策变量的下界（标量或向量）
% 输出:
%       Positions        - 初始化后的种群位置矩阵，每列代表一个个体，每行代表一个决策变量
% 作者: [您的名字]
% 日期: [日期]
% -------------------------------------------------------------

function Positions = initialization(SearchAgents_no, dim, ub, lb)
    % 计算边界的数量，判断上界和下界是标量还是向量
    Boundary_no = size(ub, 2); % 边界的数量
    
    % 如果所有变量的上下界相同，且用户输入的是单个数值
    if Boundary_no == 1
        % 将上界和下界扩展为与决策变量维度相同的向量
        ub_new = ones(1, dim) * ub;
        lb_new = ones(1, dim) * lb;
    else
        % 否则，直接使用用户输入的向量作为上界和下界
        ub_new = ub;
        lb_new = lb;   
    end
    
    % 初始化种群位置矩阵
    % 每个个体的每个决策变量值在对应的上下界之间随机生成
    for i = 1:dim
        % 获取当前决策变量的上界和下界
        ub_i = ub_new(i);
        lb_i = lb_new(i);
        
        % 生成 SearchAgents_no 个个体在第 i 个决策变量上的值
        % 使用 rand 生成 [0,1] 之间的随机数，并缩放到 [lb_i, ub_i] 范围
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
    end
    
    % 转置矩阵，使得每列代表一个个体，每行代表一个决策变量
    Positions = Positions';
end
