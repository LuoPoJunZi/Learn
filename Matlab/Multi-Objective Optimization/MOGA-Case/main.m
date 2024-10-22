% 变量范围
lb = zeros(1, 10); % 下界：10个决策变量的下限均为0
ub = ones(1, 10);  % 上界：10个决策变量的上限均为1

% GA参数
% 设置遗传算法的参数
ga_options = optimoptions(@gamultiobj, ...
    'MaxGenerations', 5000, ... % 最大代数
    'PopulationSize', 2000, ...  % 种群规模
    'CrossoverFraction', 0.9, ... % 交叉概率
    'MutationFcn', {@mutationgaussian, 0.1}, ... % 变异函数和变异概率
    'SelectionFcn', {@selectiontournament, 2}); % 选择函数及其参数

% 求解最优参数
% 使用多目标遗传算法求解最优决策变量 x 和对应的适应度值 fval
[x, fval] = gamultiobj(@fitness, 10, [], [], [], [], lb, ub, ga_options);

% 输出结果
disp(['找到的Pareto前沿数量：' num2str(size(fval, 1))]); % 输出找到的帕累托前沿的数量

% 绘制帕累托前沿
plot3(-fval(:, 1), -fval(:, 2), fval(:, 3), '.'); % 绘制三维帕累托前沿
xlabel('功率'); % x轴标签
ylabel('效率'); % y轴标签
zlabel('成本'); % z轴标签
title('Pareto Front'); % 图标题

% 将适应度值转置以便后续处理
A1 = -fval(:, 1); % 功率
B1 = -fval(:, 2); % 效率
C1 = fval(:, 3);   % 成本
optpem = -fval;    % 存储优化的适应度值

% 数据，根据设计的权重找到最佳解
Y = -fval(:, 1)'; % 功率的负值
B = -fval(:, 2)'; % 效率的负值
C = fval(:, 3)';   % 成本

% 权重
weights = [0.3165, 0.3670, 0.3165]; % 权重向量，权重之和应为1

% 标准化
normA = Y / norm(Y); % 对功率进行标准化
normB = B / norm(B); % 对效率进行标准化
normC = C / norm(C); % 对成本进行标准化

% 加权标准化矩阵
weighted_normA = weights(1) * normA; % 加权后的功率标准化
weighted_normB = weights(2) * normB; % 加权后的效率标准化
weighted_normC = weights(3) * normC; % 加权后的成本标准化

% 正理想解和负理想解
idealPositive = [max(weighted_normA), max(weighted_normB), min(weighted_normC)]; % 正理想解
idealNegative = [min(weighted_normA), min(weighted_normB), max(weighted_normC)]; % 负理想解

% 计算分离度
distancePositive = sqrt((weighted_normA - idealPositive(1)).^2 + ...
    (weighted_normB - idealPositive(2)).^2 + ...
    (weighted_normC - idealPositive(3)).^2); % 到正理想解的距离

distanceNegative = sqrt((weighted_normA - idealNegative(1)).^2 + ...
    (weighted_normB - idealNegative(2)).^2 + ...
    (weighted_normC - idealNegative(3)).^2); % 到负理想解的距离

% 计算相对接近度
relativeCloseness = distanceNegative ./ (distancePositive + distanceNegative); % 相对接近度计算

% 找到最佳点
[bestValue, bestIndex] = max(relativeCloseness); % 找到相对接近度最大的点

% 输出最佳点的信息
fprintf('最佳点的索引是：%d, 相对接近度为：%.4f\n', bestIndex, bestValue); % 输出最佳点的索引和接近度

AAA = abs(fval); % 计算适应度值的绝对值
