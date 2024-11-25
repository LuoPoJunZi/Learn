function plot_data2(M, D, Pareto)
%% 绘制帕累托解集的函数
% 输入参数：
% M - 目标函数的数量
% D - 决策变量的数量
% Pareto - 帕累托解集，包括决策变量和目标函数值
% 输出：
% 在二维平面上绘制帕累托解集的散点图

%% 提取绘图数据
% 从帕累托解集中提取目标函数值
pl_data = Pareto(:, D+1:D+M); % 提取帕累托解集中所有个体的目标函数值
pl_data = sortrows(pl_data, 2); % 按第二目标值升序排序

% 分离目标值
X = pl_data(:, 1); % 第一个目标值
Y = pl_data(:, 2); % 第二个目标值

%% 绘制帕累托前沿
figure; % 创建新图形窗口
scatter(X, Y, '*', 'k'); % 用黑色星号绘制散点图

% 添加标题和坐标轴标签
title('Optimal Solution Pareto Set'); % 图表标题
xlabel('Objective function value 1'); % X轴标签
ylabel('Objective function value 2'); % Y轴标签
grid on; % 显示网格

% 如果需要进一步美化图形，例如添加颜色渐变或额外标记，可在此处扩展
end
