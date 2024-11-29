function pop=DetermineDomination(pop)
% DetermineDomination 确定种群中每个个体是否被支配
% 输入参数:
%   pop - 当前种群，包含目标值
% 输出参数:
%   pop - 更新后的种群，包含IsDominated标志

    n = numel(pop);  % 种群大小
    
    % 初始化每个个体的支配标志为false
    for i = 1:n
        pop(i).IsDominated = false;
    end
    
    % 双重循环，比较每对个体
    for i = 1:n
        if pop(i).IsDominated
            continue;  % 如果已被支配，跳过
        end
        
        for j = 1:n
            if Dominates(pop(j), pop(i))
                pop(i).IsDominated = true;  % 被j支配
                break;  % 不需要继续检查
            end
        end
    end

end
