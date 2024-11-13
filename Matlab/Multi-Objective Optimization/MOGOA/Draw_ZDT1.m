function TPF = Draw_ZDT1()
% 绘制 ZDT1 问题的真实帕累托前沿（True Pareto Front, TPF）
% 该函数绘制的是 ZDT1 问题在给定输入下的真实目标前沿。

% 定义目标函数 ZDT1
ObjectiveFunction = @(x) ZDT1(x);

% 设置 x 变量的取值范围
x = 0:0.01:1;

% 计算 ZDT1 的真实帕累托前沿
for i = 1:size(x, 2)
    TPF(i, :) = ObjectiveFunction([x(i) 0 0 0]); % 计算每个 x 对应的目标值
end

% 绘制帕累托前沿
line(TPF(:, 1), TPF(:, 2)); % 绘制目标空间中的线
title('ZDT1'); % 设置标题

% 设置坐标轴标签
xlabel('f1'); % f1 目标
ylabel('f2'); % f2 目标

% 显示坐标轴边框
box on;

% 设置图形字体
fig = gcf; % 获取当前图形
set(findall(fig, '-property', 'FontName'), 'FontName', 'Garamond'); % 设置字体为 Garamond
set(findall(fig, '-property', 'FontAngle'), 'FontAngle', 'italic'); % 设置字体斜体
