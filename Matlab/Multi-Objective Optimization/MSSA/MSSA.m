% -------------------------------------------------------------
% 文件名: MSSA.m
% 功能: 实现多目标Salp群算法（Multi-objective Salp Swarm Algorithm, MSSA）
%       该算法用于解决多目标优化问题，以ZDT1作为测试函数。
%       主要步骤包括初始化种群、评估适应度、更新食物位置、
%       维护和更新存档（Archive），并最终绘制结果。
% 输入:
%       无（所有参数在脚本中定义）
% 输出:
%       绘制ZDT1的真实前沿和MSSA获得的前沿曲线
% 作者: [您的名字]
% 日期: [日期]
% -------------------------------------------------------------

clc;            % 清除命令行窗口
clear;          % 清除工作区变量
close all;      % 关闭所有打开的图形窗口

% ==================== 参数设置 ====================

% 定义目标函数，这里使用ZDT1测试函数
ObjectiveFunction = @ZDT1;

% 决策变量的维度
dim = 5;

% 决策变量的下界和上界（可以是标量或向量）
lb = 0;
ub = 1;

% 目标函数的数量
obj_no = 2;

% 如果上界和下界是标量，则扩展为与维度相同的向量
if size(ub, 2) == 1
    ub = ones(1, dim) * ub;
    lb = ones(1, dim) * lb;
end

% 最大迭代次数
max_iter = 100;

% 种群规模（搜索代理的数量）
N = 200;

% 存档的最大容量
ArchiveMaxSize = 100;

% 初始化存档中的决策变量矩阵，预分配空间为100行dim列
Archive_X = zeros(100, dim);

% 初始化存档中的目标函数值矩阵，预分配空间为100行obj_no列，初始值为正无穷
Archive_F = ones(100, obj_no) * inf;

% 当前存档中的个体数量，初始为0
Archive_member_no = 0;

% 速度和位置的初始化参数
r = (ub - lb) / 2;                  % 速度初始化参数（未在脚本中使用）
V_max = (ub(1) - lb(1)) / 10;       % 速度的最大值（未在脚本中使用）

% 食物个体的适应度和位置初始化
Food_fitness = inf * ones(1, obj_no);    % 食物个体的目标函数值，初始为正无穷
Food_position = zeros(dim, 1);           % 食物个体的位置，初始为零向量

% 初始化Salp群的位置，生成N个个体，每个个体有dim个决策变量
Salps_X = initialization(N, dim, ub, lb);

% 初始化Salp群的适应度矩阵，存储N个个体的适应度
fitness = zeros(N, 2);

% 初始化速度矩阵（未在脚本中使用）
V = initialization(N, dim, ub, lb);

% 初始化位置历史记录矩阵（未在脚本中使用）
position_history = zeros(N, max_iter, dim);

% ==================== 迭代优化过程 ====================

% 主循环，迭代max_iter次
for iter = 1:max_iter
    
    % 计算控制参数c1，根据公式 (3.2) 在论文中定义
    c1 = 2 * exp(-(4 * iter / max_iter)^2);
    
    % 计算所有Salp个体的目标函数值
    for i = 1:N
        % 计算第i个Salp个体的目标函数值
        Salps_fitness(i, :) = ObjectiveFunction(Salps_X(:, i)');
        
        % 如果当前Salp的适应度支配食物个体的适应度，则更新食物个体
        if dominates(Salps_fitness(i, :), Food_fitness)
            Food_fitness = Salps_fitness(i, :);
            Food_position = Salps_X(:, i);
        end
    end
    
    % 更新存档，将当前Salp群体加入存档中
    [Archive_X, Archive_F, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, Salps_X, Salps_fitness, Archive_member_no);
    
    % 如果存档超过最大容量，则处理满存档的情况
    if Archive_member_no > ArchiveMaxSize
        % 对存档中的个体进行排序和排名
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
        
        % 处理满存档，移除部分个体
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
    else
        % 对存档中的个体进行排序和排名
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    end
    
    % 再次对存档中的个体进行排序和排名（冗余，可能需要优化）
    Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    
    % 使用轮盘赌选择方法，从存档中选择一个个体作为新的食物个体
    % 选择概率与个体排名的倒数成正比，以提高覆盖度
    index = RouletteWheelSelection(1 ./ Archive_mem_ranks);
    
    % 如果选择结果为-1（异常情况），则默认选择第一个个体
    if index == -1
        index = 1;
    end
    
    % 更新食物个体的适应度和位置
    Food_fitness = Archive_F(index, :);
    Food_position = Archive_X(index, :)';
    
    % 更新Salp群的位置
    for i = 1:N
        index = 0;             % 初始化索引（未使用）
        neighbours_no = 0;     % 初始化邻居数量（未使用）
        
        if i <= N / 2
            % 前半部分Salp个体更新位置
            for j = 1:dim
                c2 = rand();    % 生成一个[0,1]之间的随机数
                c3 = rand();    % 生成另一个[0,1]之间的随机数
                
                % 根据公式 (3.1) 在论文中定义，更新Salp的位置
                if c3 < 0.5
                    Salps_X(j, i) = Food_position(j) + c1 * ((ub(j) - lb(j)) * c2 + lb(j));
                else
                    Salps_X(j, i) = Food_position(j) - c1 * ((ub(j) - lb(j)) * c2 + lb(j));
                end
            end
        elseif i > N / 2 && i < N + 1
            % 后半部分Salp个体更新位置，通过前一个个体的位置和当前个体的位置的平均值
            point1 = Salps_X(:, i - 1);
            point2 = Salps_X(:, i);
            
            % 根据公式 (3.4) 在论文中定义，更新Salp的位置
            Salps_X(:, i) = (point2 + point1) / 2;
        end
        
        % 位置边界处理，确保每个决策变量在上下界内
        Flag4ub = Salps_X(:, i) > ub';
        Flag4lb = Salps_X(:, i) < lb';
        Salps_X(:, i) = (Salps_X(:, i) .* ~(Flag4ub + Flag4lb)) + ub' .* Flag4ub + lb' .* Flag4lb;
    end
    
    % 显示当前迭代的信息，包括迭代次数和存档中的非支配解数量
    disp(['At the iteration ', num2str(iter), ' there are ', num2str(Archive_member_no), ' non-dominated solutions in the archive']);
    
end

% ==================== 结果绘制 ====================

figure;                     % 创建一个新图形窗口
Draw_ZDT1();               % 绘制ZDT1的真实目标前沿
hold on;                    % 保持当前图形，便于后续绘制

% 绘制存档中的非支配解，使用红色圆圈标记，填充颜色为黑色
plot(Archive_F(:, 1), Archive_F(:, 2), 'ro', 'MarkerSize', 8, 'markerfacecolor', 'k');

% 添加图例，标明真实前沿和获得的前沿
legend('True PF', 'Obtained PF');

% 设置图形标题
title('MSSA');
