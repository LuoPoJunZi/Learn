% 函数 GetCosts
% 该函数用于从种群中提取所有个体的目标值 (Cost)，并将其组织为一个矩阵。
% 参数：
%   - pop: 种群，包含多个个体，每个个体具有一个目标值 (Cost) 向量
% 返回：
%   - costs: 一个矩阵，其中每一列对应一个个体的目标值，每一行对应一个目标

function costs = GetCosts(pop)

    % 获取每个个体的目标数量，即目标值 (Cost) 向量的维度
    nobj = numel(pop(1).Cost);

    % 将种群中的所有目标值提取并重组为一个 nobj 行的矩阵
    % 每列表示一个个体的目标值，每行表示某个目标在所有个体中的值
    costs = reshape([pop.Cost], nobj, []);

end
