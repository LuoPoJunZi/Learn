% Draw_ZDT1.m - 绘制 ZDT1 问题的真实帕累托前沿
%
% 输出:
%   绘制 ZDT1 问题的帕累托前沿图形

function TPF = Draw_ZDT1()
    % TPF 是真实的帕累托前沿
    addpath('ZDT_set'); % 添加包含 ZDT1 函数的路径

    % 定义目标函数为 ZDT1
    ObjectiveFunction = @(x) ZDT1(x);
    
    % 在 x 轴上生成从 0 到 1 的点
    x = 0:0.01:1;
    
    % 计算每个 x 对应的目标值
    for i = 1:length(x)
        TPF(i, :) = ObjectiveFunction([x(i) 0 0 0]);
    end
    
    % 绘制帕累托前沿
    line(TPF(:, 1), TPF(:, 2), 'LineWidth', 2);
    title('ZDT1');
    xlabel('f1');
    ylabel('f2');
    box on;
    
    % 设置图形的字体和样式
    fig = gcf;
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Garamond');
    set(findall(fig, '-property', 'FontAngle'), 'FontAngle', 'italic');
end
