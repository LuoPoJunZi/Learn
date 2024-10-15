function pop=CalcCrowdingDistance(pop,F)
    % 计算种群中每个个体的拥挤度，用于NSGA-II算法中的拥挤度比较操作
    
    nF = numel(F);  % 获取非支配前沿的数量
    
    for k = 1:nF  % 遍历每一个非支配前沿
        
        Costs = [pop(F{k}).Cost];  % 获取当前非支配前沿个体的目标函数值
        
        nObj = size(Costs, 1);  % 目标函数的个数
        
        n = numel(F{k});  % 当前非支配前沿中个体的数量
        
        d = zeros(n, nObj);  % 初始化拥挤度矩阵
        
        for j = 1:nObj  % 对每个目标函数计算拥挤度
            
            [cj, so] = sort(Costs(j,:));  % 按照第j个目标函数对个体进行排序
            
            d(so(1), j) = inf;  % 最边界的个体的拥挤度设为无穷大
            
            for i = 2:n-1
                % 计算第i个个体在第j个目标上的拥挤度
                d(so(i), j) = abs(cj(i+1) - cj(i-1)) / abs(cj(1) - cj(end));
                % 公式： (下一邻居目标值 - 上一邻居目标值) / (最大目标值 - 最小目标值)
            end
            
            d(so(end), j) = inf;  % 另一端边界个体的拥挤度设为无穷大
            
        end
        
        % 计算每个个体的总体拥挤度
        for i = 1:n
            pop(F{k}(i)).CrowdingDistance = sum(d(i,:));  % 拥挤度是对各目标上的拥挤度求和
        end
        
    end

end
