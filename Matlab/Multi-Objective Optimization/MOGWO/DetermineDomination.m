% 函数 DetermineDomination
% 该函数用于确定种群中每个解的支配关系，即判断每个个体是否被其他个体支配。
% 这一步对于多目标优化非常重要，因为它决定了哪些解是非支配的，哪些解将被淘汰。
% 参数：
%   - pop: 种群，包含多个个体，每个个体有自己的目标值和支配状态
% 返回：
%   - pop: 更新后的种群，其中每个个体的支配状态（Dominated）已确定

function pop = DetermineDomination(pop)

    % 获取种群中个体的数量
    npop = numel(pop);
    
    % 遍历种群中的每个个体 i
    for i = 1:npop
        % 首先假设个体 i 未被支配
        pop(i).Dominated = false;
        
        % 与种群中的其他个体 j 进行比较（只比较 i 之前的个体）
        for j = 1:i-1
            % 只有当个体 j 未被支配时，才需要进行支配关系的检查
            if ~pop(j).Dominated
                % 如果个体 i 支配个体 j，则标记 j 被支配
                if Dominates(pop(i), pop(j))
                    pop(j).Dominated = true;
                % 如果个体 j 支配个体 i，则标记 i 被支配，并停止进一步比较
                elseif Dominates(pop(j), pop(i))
                    pop(i).Dominated = true;
                    break;  % 一旦 i 被支配，不再需要与其他个体比较
                end
            end
        end
    end

end
