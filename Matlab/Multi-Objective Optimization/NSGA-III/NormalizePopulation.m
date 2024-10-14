function [pop, params] = NormalizePopulation(pop, params)
    % NormalizePopulation 函数用于对种群进行归一化处理
    % 输入：
    %   pop - 当前种群（个体集合）
    %   params - 包含优化参数和状态的结构体
    % 输出：
    %   pop - 更新后的种群，包含归一化后的成本
    %   params - 更新后的参数结构体

    % 更新理想点 zmin
    params.zmin = UpdateIdealPoint(pop, params.zmin);
    
    % 计算每个个体的成本相对于理想点的偏差
    fp = [pop.Cost] - repmat(params.zmin, 1, numel(pop));
    
    % 使用标量化方法更新参数
    params = PerformScalarizing(fp, params);
    
    % 计算超平面的截距
    a = FindHyperplaneIntercepts(params.zmax);
    
    % 对每个个体进行归一化处理
    for i = 1:numel(pop)
        pop(i).NormalizedCost = fp(:,i) ./ a;  % 归一化成本
    end
end

function a = FindHyperplaneIntercepts(zmax)
    % FindHyperplaneIntercepts 函数用于计算超平面的截距
    % 输入：
    %   zmax - 当前种群中每个目标的最大成本
    % 输出：
    %   a - 超平面的截距
    
    w = ones(1, size(zmax, 2)) / zmax;  % 计算每个目标的权重
    a = (1 ./ w)';  % 计算超平面的截距
end
