function PlotCosts(pop)

    % 从种群中提取成本（目标函数值）
    Costs = [pop.Cost];  % 将每个个体的成本值组合成一个矩阵
    
    % 绘制目标函数值的散点图
    plot(Costs(1, :), Costs(2, :), 'r*', 'MarkerSize', 8);  % 以红色星形标记绘制
    xlabel('1st Objective');  % x 轴标签
    ylabel('2nd Objective');  % y 轴标签
    grid on;  % 开启网格
    
end
