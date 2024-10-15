function [pop, F] = SortPopulation(pop)
    % 对种群进行排序
    % 输入：
    %   pop - 种群（个体集合），每个个体包含其拥挤距离和支配等级
    % 输出：
    %   pop - 排序后的种群
    %   F - 每个等级对应的个体索引

    % 基于拥挤距离进行排序
    [~, CDSO] = sort([pop.CrowdingDistance], 'descend');  % 降序排序
    pop = pop(CDSO);  % 按照拥挤距离排序后的顺序更新种群
    
    % 基于支配等级进行排序
    [~, RSO] = sort([pop.Rank]);  % 升序排序
    pop = pop(RSO);  % 按照支配等级排序后的顺序更新种群
    
    % 更新前沿集合 (Update Fronts)
    Ranks = [pop.Rank];  % 获取排序后的个体的支配等级
    MaxRank = max(Ranks);  % 确定最大的支配等级
    F = cell(MaxRank, 1);  % 初始化每个等级对应的个体索引的单元格数组
    
    for r = 1:MaxRank
        F{r} = find(Ranks == r);  % 找到每个等级对应的个体索引
    end
end
