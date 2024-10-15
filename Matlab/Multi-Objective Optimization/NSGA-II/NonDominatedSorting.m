function [pop, F] = NonDominatedSorting(pop)
    % 非支配排序函数，用于NSGA-II算法中的非支配排序操作
    % 输入：
    %   pop - 种群（个体集合），包含个体的目标函数值和支配信息
    % 输出：
    %   pop - 更新后的种群，包含个体的支配集和支配计数信息
    %   F - 各个非支配前沿的个体索引集合

    nPop = numel(pop);  % 种群中个体的数量

    % 初始化每个个体的支配集（DominationSet）和被支配计数（DominatedCount）
    for i = 1:nPop
        pop(i).DominationSet = [];  % 初始化支配集为空
        pop(i).DominatedCount = 0;  % 初始化被支配计数为0
    end
    
    F{1} = [];  % 初始化第一个非支配前沿

    % 进行两两个体比较，确定支配关系
    for i = 1:nPop
        for j = i + 1:nPop
            p = pop(i);
            q = pop(j);
            
            % 判断个体p是否支配个体q
            if Dominates(p, q)
                p.DominationSet = [p.DominationSet j];  % p支配q，q加入p的支配集
                q.DominatedCount = q.DominatedCount + 1;  % q的被支配计数加1
            end
            
            % 判断个体q是否支配个体p
            if Dominates(q.Cost, p.Cost)
                q.DominationSet = [q.DominationSet i];  % q支配p，p加入q的支配集
                p.DominatedCount = p.DominatedCount + 1;  % p的被支配计数加1
            end
            
            pop(i) = p;  % 更新个体p
            pop(j) = q;  % 更新个体q
        end
        
        % 如果个体i没有被其他个体支配，则它属于第一个非支配前沿
        if pop(i).DominatedCount == 0
            F{1} = [F{1} i];  % 将个体i加入第一个非支配前沿
            pop(i).Rank = 1;  % 设置该个体的Rank为1
        end
    end
    
    k = 1;  % 当前非支配前沿的索引

    % 迭代生成后续非支配前沿
    while true
        
        Q = [];  % 临时数组，用于存储当前非支配前沿的个体
        
        % 遍历当前非支配前沿的所有个体
        for i = F{k}
            p = pop(i);
            
            % 遍历p支配的个体
            for j = p.DominationSet
                q = pop(j);
                
                q.DominatedCount = q.DominatedCount - 1;  % 被支配计数减1
                
                % 如果q的被支配计数为0，说明它属于下一非支配前沿
                if q.DominatedCount == 0
                    Q = [Q j];  % 将q加入临时数组Q
                    q.Rank = k + 1;  % 设置q的Rank为当前前沿的下一层
                end
                
                pop(j) = q;  % 更新个体q
            end
        end
        
        % 如果Q为空，说明所有个体已经完成排序，跳出循环
        if isempty(Q)
            break;
        end
        
        F{k + 1} = Q;  % 将Q作为新的非支配前沿
        k = k + 1;  % 递增前沿索引
        
    end

end
