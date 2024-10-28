clc;
clear;
close all;
format long g

% 初始化 MODA (Multi-Objective Multi-Verse Optimization Algorithm) 算法的参数
max_iter = 100;       % 最大迭代次数
N = 100;              % 搜索代理（个体）数量
ArchiveMaxSize = 100; % 归档的最大容量
obj_no = 2;           % 目标函数的数量

% 载入测试数据，用于绘制真实帕累托前沿
Archive_F1 = load('weldedbeam.txt'); 

% 运行 MOMVO 算法，得到最优解的分数、位置和归档的目标函数值
[Best_universe_score, Best_universe_pos, Archive_F] = MOMVO(max_iter, N, ArchiveMaxSize);

% 绘制真实帕累托前沿（从文件中读取）
plot(Archive_F1(:, 1), Archive_F1(:, 2), 'Color', 'b', 'LineWidth', 4);
hold on

% 绘制 MOMVO 算法得到的帕累托前沿
plot(Archive_F(:, 1), Archive_F(:, 2), 'ro', 'LineWidth', 2, ...
    'MarkerEdgeColor', 'r', ...
    'MarkerFaceColor', 'r', ...
    'MarkerSize', 6);

% 添加图例、标题及坐标标签
legend('True PF', 'Obtained PF');
title('MOMVO FOR Welded Beam Design PROBLEM');
xlabel('obj_1'); % 横坐标表示第一个目标函数
ylabel('obj_2'); % 纵坐标表示第二个目标函数
