function params = PerformScalarizing(z, params)

    nObj = size(z, 1);  % 目标函数数量
    nPop = size(z, 2);  % 种群大小
    
    % 如果 smin 参数不为空，使用已有的 zmax 和 smin
    if ~isempty(params.smin)
        zmax = params.zmax;  % 记录当前的最优点
        smin = params.smin;  % 记录当前的最小标量值
    else
        zmax = zeros(nObj, nObj);  % 初始化最优点为零
        smin = inf(1, nObj);  % 初始化最小标量值为无穷大
    end
    
    % 对每个目标函数进行标量化处理
    for j = 1:nObj
       
        w = GetScalarizingVector(nObj, j);  % 获取标量化向量
        
        s = zeros(1, nPop);  % 初始化标量值数组
        for i = 1:nPop
            % 计算每个个体的标量值
            s(i) = max(z(:, i) ./ w);
        end

        [sminj, ind] = min(s);  % 找到当前标量值中的最小值及其索引
        
        % 如果当前最小标量值小于记录的最小值，则更新 zmax 和 smin
        if sminj < smin(j)
            zmax(:, j) = z(:, ind);  % 更新最优点
            smin(j) = sminj;  % 更新最小标量值
        end
        
    end
    
    % 将更新后的 zmax 和 smin 存回 params 结构体中
    params.zmax = zmax;
    params.smin = smin;
    
end

function w = GetScalarizingVector(nObj, j)

    epsilon = 1e-10;  % 设定一个很小的值，以避免除零

    w = epsilon * ones(nObj, 1);  % 初始化标量化向量

    w(j) = 1;  % 将第 j 个目标的权重设置为 1

end
