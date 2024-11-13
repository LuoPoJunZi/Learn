clc;
clear;
close all;

% 根据问题的具体情况设置以下参数
ObjectiveFunction = @ZDT1;  % 目标函数（此处为ZDT1）
dim = 5;  % 搜索空间的维度
lb = 0;  % 下边界
ub = 1;  % 上边界
obj_no = 2;  % 目标函数数量

% 如果上下边界是单一的数值，扩展为与维度相同的向量
if size(ub, 2) == 1
    ub = ones(1, dim) * ub;
    lb = ones(1, dim) * lb;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 如果维度是奇数，则增加一个维度，确保维度为偶数
flag = 0;
if (rem(dim, 2) ~= 0)
    dim = dim + 1;  % 增加维度
    ub = [ub, 1];  % 增加上边界
    lb = [lb, 0];  % 增加下边界
    flag = 1;
end

% 设置迭代次数、种群大小和存档的最大容量
max_iter = 100;
N = 200;  % 搜索代理数量（即草hopper数量）
ArchiveMaxSize = 100;

% 初始化存档
Archive_X = zeros(100, dim);
Archive_F = ones(100, obj_no) * inf;
Archive_member_no = 0;

% 初始化人工草hopper的位置
GrassHopperPositions = initialization(N, dim, ub, lb);

TargetPosition = zeros(dim, 1);  % 目标位置
TargetFitness = inf * ones(1, obj_no);  % 初始目标适应度

% 设置常数值
cMax = 1;
cMin = 0.00004;

% 计算初始草hopper位置的适应度
for iter = 1:max_iter
    for i = 1:N
        % 限制草hopper的位置在边界内
        Flag4ub = GrassHopperPositions(:, i) > ub';
        Flag4lb = GrassHopperPositions(:, i) < lb';
        GrassHopperPositions(:, i) = (GrassHopperPositions(:, i) .* (~(Flag4ub + Flag4lb))) + ub' .* Flag4ub + lb' .* Flag4lb;
        
        % 计算草hopper的适应度
        GrassHopperFitness(i, :) = ObjectiveFunction(GrassHopperPositions(:, i)');
        
        % 如果当前草hopper的适应度优于目标适应度，则更新目标位置和适应度
        if dominates(GrassHopperFitness(i, :), TargetFitness)
            TargetFitness = GrassHopperFitness(i, :);
            TargetPosition = GrassHopperPositions(:, i);
        end
    end
    
    % 更新存档
    [Archive_X, Archive_F, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, GrassHopperPositions, GrassHopperFitness, Archive_member_no);
    
    % 如果存档已满，处理存档并进行选择
    if Archive_member_no > ArchiveMaxSize
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
    else
        Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    end
    
    % 使用轮盘赌选择算法选择目标位置
    Archive_mem_ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    index = RouletteWheelSelection(1 ./ Archive_mem_ranks);
    if index == -1
        index = 1;
    end
    TargetFitness = Archive_F(index, :);
    TargetPosition = Archive_X(index, :)';
    
    % 计算调整系数 c，随迭代次数减小
    c = cMax - iter * ((cMax - cMin) / max_iter);  % Eq. (3.8) 公式

    % 更新草hopper的位置
    for i = 1:N
        temp = GrassHopperPositions;
        
        for k = 1:2:dim
            S_i = zeros(2, 1);
            for j = 1:N
                if i ~= j
                    Dist = distance(temp(k:k+1, j), temp(k:k+1, i));
                    r_ij_vec = (temp(k:k+1, j) - temp(k:k+1, i)) / (Dist + eps);
                    xj_xi = 2 + rem(Dist, 2);
                    
                    % 计算相互作用项（参考论文 Eq. (3.2)）
                    s_ij = ((ub(k:k+1)' - lb(k:k+1)') .* c / 2) * S_func(xj_xi) .* r_ij_vec;
                    S_i = S_i + s_ij;
                end
            end
            S_i_total(k:k+1, :) = S_i;
        end
        
        % 计算新的位置（参考论文 Eq. (3.7)）
        X_new = c * S_i_total' + (TargetPosition)';  
        GrassHopperPositions_temp(i, :) = X_new';
    end
    
    % 更新草hopper的位置
    GrassHopperPositions = GrassHopperPositions_temp';
    
    % 打印当前迭代的信息
    disp(['At the iteration ', num2str(iter), ' there are ', num2str(Archive_member_no), ' non-dominated solutions in the archive']);
end

% 如果维度是奇数，去掉最后一维
if (flag == 1)
    TargetPosition = TargetPosition(1:dim-1);
end

% 绘制结果
figure
Draw_ZDT1();  % 绘制真实帕累托前沿

hold on

% 绘制获得的帕累托前沿
plot(Archive_F(:, 1), Archive_F(:, 2), 'ro', 'MarkerSize', 8, 'markerfacecolor', 'k');

legend('True PF', 'Obtained PF');
title('MOGOA');

% 设置图形窗口位置
set(gcf, 'pos', [403 466 230 200])
