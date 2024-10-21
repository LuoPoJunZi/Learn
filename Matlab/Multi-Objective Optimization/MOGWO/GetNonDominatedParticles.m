% 函数 GetNonDominatedParticles
% 该函数用于从种群中提取非支配粒子，这些粒子在多目标优化中是最优解集的一部分。
% 参数：
%   - pop: 种群，包含多个个体，每个个体都有一个布尔属性 Dominated
% 返回：
%   - nd_pop: 非支配粒子的集合，即不被其他粒子支配的粒子

function nd_pop = GetNonDominatedParticles(pop)

    % 使用逻辑索引提取未被支配的粒子
    % ND 是一个布尔数组，表示哪些粒子未被其他粒子支配
    ND = ~[pop.Dominated];
    
    % 根据 ND 的布尔值提取非支配粒子
    nd_pop = pop(ND);

end
