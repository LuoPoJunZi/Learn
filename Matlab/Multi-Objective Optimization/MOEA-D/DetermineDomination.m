% 确定种群中个体的支配关系的函数
% 输入参数:
% pop - 种群数组，包含多个个体

function pop = DetermineDomination(pop)

    nPop = numel(pop);  % 获取种群中个体的数量

    % 初始化每个个体的支配状态
    for i = 1:nPop
        pop(i).IsDominated = false;  % 默认未被支配
    end
    
    % 进行双重循环比较每对个体
    for i = 1:nPop
        for j = i + 1:nPop
            if Dominates(pop(i), pop(j))
                % 如果个体i支配个体j
                pop(j).IsDominated = true;  % 将j标记为被支配
                
            elseif Dominates(pop(j), pop(i))
                % 如果个体j支配个体i
                pop(i).IsDominated = true;  % 将i标记为被支配
                
            end
        end
    end

end
