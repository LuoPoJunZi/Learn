% 该函数初始化搜索代理的第一代种群
function Positions = initialization(SearchAgents_no, dim, ub, lb)
% 输入:
%   SearchAgents_no: 搜索代理的数量
%   dim: 每个搜索代理的维度
%   ub: 每个变量的上边界（可以是单一数值或向量）
%   lb: 每个变量的下边界（可以是单一数值或向量）
% 输出:
%   Positions: 初始化的搜索代理位置矩阵，每个行表示一个搜索代理的坐标

% 获取边界的数量
Boundary_no = size(ub, 2); % 边界的数量

% 如果所有变量的边界相同且用户输入的是单个上边界和下边界值
if Boundary_no == 1
    ub_new = ones(1, dim) * ub;  % 将单一的上边界扩展为与维度相同的向量
    lb_new = ones(1, dim) * lb;  % 将单一的下边界扩展为与维度相同的向量
else
    % 否则直接使用用户输入的上边界和下边界
    ub_new = ub;
    lb_new = lb;   
end

% 初始化每个搜索代理的位置
for i = 1:dim
    % 获取第 i 个维度的上边界和下边界
    ub_i = ub_new(i);
    lb_i = lb_new(i);
    
    % 在该维度范围内随机生成位置
    Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
end

% 转置 Positions，使得每一行对应一个搜索代理的位置
Positions = Positions';
