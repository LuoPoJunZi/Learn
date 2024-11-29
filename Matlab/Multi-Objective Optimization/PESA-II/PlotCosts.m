function PlotCosts(PF)
% PlotCosts 绘制Pareto前沿的目标值
% 输入参数:
%   PF - Pareto前沿的个体数组，包含Cost字段

    % 提取所有个体的目标值
    PFC = [PF.Cost];
    
    % 绘制第一个目标与第二个目标的关系
    plot(PFC(1, :), PFC(2, :), 'x');
    xlabel('1^{st} Objective');  % x轴标签
    ylabel('2^{nd} Objective');  % y轴标签
    grid on;                     % 显示网格

end
