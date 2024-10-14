function [pop, F, params] = SortAndSelectPopulation(pop, params)

    % 归一化种群
    [pop, params] = NormalizePopulation(pop, params);

    % 非支配排序
    [pop, F] = NonDominatedSorting(pop);
    
    nPop = params.nPop;  % 获取种群大小
    if numel(pop) == nPop
        return;  % 如果当前种群数量等于预设数量，则直接返回
    end
    
    % 关联参考点
    [pop, d, rho] = AssociateToReferencePoint(pop, params);
    
    newpop = [];  % 初始化新种群
    for l = 1:numel(F)
        % 如果当前新种群加上当前前沿超出种群大小
        if numel(newpop) + numel(F{l}) > nPop
            LastFront = F{l};  % 记录当前前沿
            break;  % 退出循环
        end
        
        newpop = [newpop; pop(F{l})];   %#ok 添加当前前沿的个体到新种群
    end
    
    while true
        % 找到最小 rho 值的索引 j
        [~, j] = min(rho);
        
        AssocitedFromLastFront = [];  % 记录从最后一个前沿关联的个体
        for i = LastFront
            if pop(i).AssociatedRef == j
                AssocitedFromLastFront = [AssocitedFromLastFront i]; %#ok
            end
        end
        
        % 如果没有与 j 关联的个体，设置 rho(j) 为无穷大
        if isempty(AssocitedFromLastFront)
            rho(j) = inf;
            continue;  % 继续下一个循环
        end
        
        % 如果 rho(j) 为 0，选择距离最近的个体
        if rho(j) == 0
            ddj = d(AssocitedFromLastFront, j);
            [~, new_member_ind] = min(ddj);  % 找到最小距离的个体
        else
            % 否则随机选择一个个体
            new_member_ind = randi(numel(AssocitedFromLastFront));
        end
        
        MemberToAdd = AssocitedFromLastFront(new_member_ind);  % 选择要添加的个体
        
        % 从最后前沿中移除选择的个体
        LastFront(LastFront == MemberToAdd) = [];
        
        newpop = [newpop; pop(MemberToAdd)]; %#ok 添加个体到新种群
        
        rho(j) = rho(j) + 1;  % 更新 rho
        
        % 如果新种群达到预设大小，则退出循环
        if numel(newpop) >= nPop
            break;
        end
    end
    
    % 对新种群进行非支配排序
    [pop, F] = NonDominatedSorting(newpop);
    
end
