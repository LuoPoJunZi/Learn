function P = SelectFromPopulation(pop, grid, beta)
% SelectFromPopulation 从种群中选择一个个体，基于网格和选择参数beta
% 输入参数:
%   pop - 当前种群
%   grid - 网格结构数组
%   beta - 选择操作的参数
% 输出参数:
%   P - 被选择的个体

    % 筛选出非空网格
    sg = grid([grid.N] > 0);
    
    % 计算每个网格的选择概率，基于网格中个体数量的倒数的beta次方
    p = 1 ./ [sg.N].^beta;
    p = p / sum(p);  % 归一化概率
    
    % 使用轮盘赌选择一个网格
    k = RouletteWheelSelection(p);
    
    % 获取被选网格中的成员索引
    Members = sg(k).Members;
    
    % 从成员中随机选择一个个体
    i = Members(randi([1, numel(Members)]));
    
    % 返回被选择的个体
    P = pop(i);

end
