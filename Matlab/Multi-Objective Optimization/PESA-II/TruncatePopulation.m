function [pop, grid] = TruncatePopulation(pop, grid, E, beta)
% TruncatePopulation 截断种群以满足存档大小限制
% 输入参数:
%   pop - 当前存档
%   grid - 网格结构数组
%   E - 需要删除的个体数量
%   beta - 删除操作的参数
% 输出参数:
%   pop - 更新后的存档
%   grid - 更新后的网格

    ToBeDeleted = [];  % 初始化待删除个体的索引列表
    
    for e = 1:E
        % 筛选出非空网格
        sg = grid([grid.N] > 0);
        
        % 计算每个网格的选择概率，基于网格中个体数量的beta次方
        p = [sg.N].^beta;
        p = p / sum(p);  % 归一化概率
        
        % 使用轮盘赌选择一个网格
        k = RouletteWheelSelection(p);
        
        % 获取被选网格中的成员索引
        Members = sg(k).Members;
        
        % 从成员中随机选择一个个体进行删除
        i = Members(randi([1, numel(Members)]));
        
        % 移除被删除的个体索引
        Members(Members == i) = [];
        
        % 更新网格中的成员列表和个体数量
        grid(sg(k).Index).Members = Members;
        grid(sg(k).Index).N = numel(Members);
        
        % 记录待删除的个体索引
        ToBeDeleted = [ToBeDeleted, i]; %#ok

    end
    
    % 从存档中删除选定的个体
    pop(ToBeDeleted) = [];
    
    % 注释信息（保留版权信息）

end
