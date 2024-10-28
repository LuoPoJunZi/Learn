function X = initialization(SearchAgents_no, dim, ub, lb)
% initialization 函数用于初始化搜索代理的位置（即候选解）
% 输入：
%   SearchAgents_no - 搜索代理（候选解）的数量
%   dim - 决策变量的维度
%   ub - 决策变量的上边界，可以是一个数值或向量
%   lb - 决策变量的下边界，可以是一个数值或向量
% 输出：
%   X - 初始化后的候选解矩阵，每行表示一个解，每列表示一个决策变量

Boundary_no = size(ub, 2); % 判断边界的数量

% 如果所有变量的边界相同，且用户提供了单一的上、下界数值
if Boundary_no == 1
    % 将上、下界扩展为与维度相同的向量
    ub_new = ones(1, dim) * ub;
    lb_new = ones(1, dim) * lb;
else
    % 否则直接使用给定的上、下界
    ub_new = ub;
    lb_new = lb;
end

% 如果每个变量有不同的上下界
for i = 1:dim
    ub_i = ub_new(i); % 当前变量的上界
    lb_i = lb_new(i); % 当前变量的下界
    % 在 [lb_i, ub_i] 范围内随机初始化每个搜索代理的位置
    X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
end

% 输出初始化的解矩阵 X
X = X;

end
