function PlotCosts(pop)
    % 绘制非支配解的目标函数值
    % 输入：
    %   pop - 种群（个体集合），包含每个个体的目标函数值

    Costs = [pop.Cost];  % 提取种群中所有个体的目标函数值

    % 绘制目标函数值
    plot(Costs(1,:), Costs(2,:), 'r*', 'MarkerSize', 8);
    xlabel('1^{st} Objective');  % X轴标签：第一个目标
    ylabel('2^{nd} Objective');   % Y轴标签：第二个目标
    title('Non-dominated Solutions (F_{1})');  % 图表标题
    grid on;  % 显示网格
end
