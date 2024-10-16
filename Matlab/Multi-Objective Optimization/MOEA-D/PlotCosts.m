% 绘制目标函数代价的函数
% 输入参数:
% EP - 包含非支配个体的数组

function PlotCosts(EP)

    EPC = [EP.Cost];  % 提取所有非支配个体的代价值
    plot(EPC(1, :), EPC(2, :), 'x');  % 绘制目标函数的散点图
    xlabel('1^{st} Objective');  % 设置x轴标签
    ylabel('2^{nd} Objective');  % 设置y轴标签
    grid on;  % 显示网格
    
end
