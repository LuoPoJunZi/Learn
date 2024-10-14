function [pop, F] = NonDominatedSorting(pop)
    % NonDominatedSorting 函数用于对种群进行非支配排序
    % 输入：
    %   pop - 当前种群（个体集合）
    % 输出：
    %   pop - 更新后的种群，每个个体包含其支配关系信息
    %   F - 包含每个等级（前沿）的个体索引，F{1} 是第一前沿，F{2} 是第二前沿，以此类推

    nPop = numel(pop);  % 获取种群中个体的数量

    % 初始化每个个体的支配集和被支配计数
    for i = 1:nPop
        pop(i).DominationSet = [];  % 该个体支配的个体集合
        pop(i).DominatedCount = 0;   % 该个体被支配的数量
    end
    
    F{1} = [];  % 第一前沿初始化为空
    
    % 对每对个体进行比较以建立支配关系
    for i = 1:nPop
        for j = i + 1:nPop
            p = pop(i);  % 当前个体 p
            q = pop(j);  % 当前个体 q
            
            % 如果 p 支配 q
            if Dominates(p, q)
                p.DominationSet = [p.DominationSet j];  % p 支配的个体集合增加 q
                q.DominatedCount = q.DominatedCount + 1;  % q 被支配计数加 1
            end
            
            % 如果 q 支配 p
            if Dominates(q.Cost, p.Cost)
                q.DominationSet = [q.DominationSet i];  % q 支配的个体集合增加 p
                p.DominatedCount = p.DominatedCount + 1;  % p 被支配计数加 1
            end
            
            pop(i) = p;  % 更新个体 p
            pop(j) = q;  % 更新个体 q
        end
        
        % 如果个体 p 没有被其他个体支配，则将其加入第一前沿
        if pop(i).DominatedCount == 0
            F{1} = [F{1} i];  % 将个体 i 加入第一前沿
            pop(i).Rank = 1;  % 设置个体的等级为 1
        end
    end
    
    k = 1;  % 计数器初始化为 1
    
    while true
        Q = [];  % 当前处理的前沿的个体集合
        
        % 遍历当前前沿 F{k}
        for i = F{k}
            p = pop(i);  % 当前前沿的个体 p
            
            % 遍历 p 的支配集中的每个个体
            for j = p.DominationSet
                q = pop(j);  % 被 p 支配的个体 q
                
                q.DominatedCount = q.DominatedCount - 1;  % 被支配计数减 1
                
                % 如果 q 被支配计数为 0，说明 q 进入下一前沿
                if q.DominatedCount == 0
                    Q = [Q j];  % 将个体 j 加入下一个前沿
                    q.Rank = k + 1;  % 设置个体的等级为 k + 1
                end
                
                pop(j) = q;  % 更新个体 q
            end
        end
        
        % 如果当前前沿 Q 为空，结束循环
        if isempty(Q)
            break;
        end
        
        F{k + 1} = Q;  % 将 Q 加入下一个前沿
        k = k + 1;  % 前沿计数器加 1
    end
end
