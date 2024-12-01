% PlotCosts.m
% 绘制Pareto前沿的函数

function PlotCosts(PF)
    % 输入参数:
    % PF - Pareto前沿的个体集合，包含Cost字段

    PFC = [PF.Cost];            % 提取所有个体的目标函数值
    plot(PFC(1, :), PFC(2, :), 'x');   % 绘制目标函数1与目标函数2的散点图
    xlabel('第一个目标');               % 设置x轴标签
    ylabel('第二个目标');               % 设置y轴标签
    grid on;                         % 显示网格
end
