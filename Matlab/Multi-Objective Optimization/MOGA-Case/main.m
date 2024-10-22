% main.m 文件
% 本脚本用于使用遗传算法 (GA) 求解多目标优化问题并绘制帕累托前沿

% 变量范围
lb = zeros(1, 10); % 变量的下界，10个变量均为0
ub = ones(1, 10); % 变量的上界，10个变量均为1

% GA参数设置
ga_options = optimoptions(@gamultiobj, 'PopulationSize', 500, 'MaxGenerations', 100);
% 设置种群规模为500，最大代数为100

% 求解最优参数
[x, fval] = gamultiobj(@fitness, 10, [], [], [], [], lb, ub, ga_options);
% 调用多目标遗传算法，优化目标是 fitness 函数，变量个数为10

% 输出结果
disp(['找到的Pareto前沿数量：' num2str(size(fval, 1))]);
% 显示找到的帕累托前沿数量

% 绘制帕累托前沿
plot(-fval(:, 1), -fval(:, 2), '.'); % 绘制第一和第二目标函数的负值
xlabel('功率密度'); % X轴标签
ylabel('火用效率'); % Y轴标签
title('Pareto Front'); % 图表标题
A1 = -fval(:, 1); % 保存功率密度
B1 = -fval(:, 2); % 保存火用效率
optpem = -fval; % 优化结果

% 数据，根据设计的权重找到最佳解
Y = -fval(:, 1)'; % 功率密度的负值
B = -fval(:, 2)'; % 火用效率的负值

% 权重
weights = [0.5, 0.5]; % 各目标的权重设置

% 标准化
normA = Y / norm(Y); % 功率密度的标准化
normB = B / norm(B); % 火用效率的标准化

% 加权标准化矩阵
weighted_normA = weights(1) * normA; % 加权功率密度
weighted_normB = weights(2) * normB; % 加权火用效率

% 正理想解和负理想解
idealPositive = [max(weighted_normA), min(weighted_normB)]; % 正理想解
idealNegative = [min(weighted_normA), max(weighted_normB)]; % 负理想解

% 计算分离度
distancePositive = sqrt((weighted_normA - idealPositive(1)).^2 + (weighted_normB - idealPositive(2)).^2);
% 从正理想解的距离
distanceNegative = sqrt((weighted_normA - idealNegative(1)).^2 + (weighted_normB - idealNegative(2)).^2);
% 从负理想解的距离

% 计算相对接近度
relativeCloseness = distanceNegative ./ (distancePositive + distanceNegative);
% 计算相对接近度

% 最佳点
[bestValue, bestIndex] = max(relativeCloseness); % 找到相对接近度最大的点

% 输出最佳点的信息
fprintf('最佳点的索引是：%d, 相对接近度为：%.4f\n', bestIndex, bestValue);
