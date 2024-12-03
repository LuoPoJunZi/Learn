% NSDBO.m
% 非支配排序蜣螂优化算法（NSDBO）的主函数
% 该函数初始化种群，执行优化过程，并计算评价指标
% 输入:
%   params   - 算法参数结构体，包含种群规模、仓库规模、最大代数等
%   MultiObj - 多目标优化问题的信息结构体，包括目标函数、变量范围等
% 输出:
%   f - 最终仓库中的个体，包含决策变量和目标函数值

function f = NSDBO(params, MultiObj)
    tic  % 开始计时
    
    % 获取问题名称、目标函数个数、决策变量个数及变量范围
    name = MultiObj.name;
    numOfObj = MultiObj.numOfObj;  % 目标函数个数
    evaluate_objective = MultiObj.fun;  % 目标函数句柄
    D = MultiObj.nVar;  % 决策变量维数
    LB = MultiObj.var_min;  % 决策变量下界
    UB = MultiObj.var_max;  % 决策变量上界
    
    % 获取算法参数
    Max_iteration = params.maxgen;  % 最大代数
    SearchAgents_no = params.Np;  % 种群规模
    ishow = 1;  % 显示频率
    Nr = params.Nr;  % 仓库规模
    
    % 初始化种群（决策变量和目标函数值）
    chromosome = initialize_variables(SearchAgents_no, numOfObj, D, LB, UB, evaluate_objective);
    
    % 对初始种群进行非支配排序
    intermediate_chromosome = non_domination_sort_mod(chromosome, numOfObj, D);
    
    % 替换仓库中的个体
    Pop = replace_chromosome(intermediate_chromosome, numOfObj, D, Nr);
    
    M = numOfObj;  % 目标函数个数
    K = D + M;  % 决策变量和目标函数的总维数
    
    % 提取种群的决策变量和目标函数值
    POS = Pop(:, 1:K+1);  % 包含排名的种群矩阵
    POS_ad = POS(:, 1:K);  % 适应度存储，用于更新位置
    
    % 初始化新位置矩阵
    newPOS = zeros(SearchAgents_no, K);
    
    % 检查种群中的支配关系
    DOMINATED = checkDomination(POS(:, D+1:D+M));
    
    % 移除被支配的个体，更新仓库
    Pop = POS(~DOMINATED, :);
    ERP = Pop(:, 1:K+1);  % 仓库中的个体
    
    %% 优化循环
    Iteration = 1;  % 当前代数初始化
    
    % 根据种群规模计算不同阶段的个体数量
    pNum1 = floor(SearchAgents_no * 0.2);  % 20%的个体
    pNum2 = floor(SearchAgents_no * 0.4);  % 40%的个体
    pNum3 = floor(SearchAgents_no * 0.63); % 63%的个体
    
    while Iteration <= Max_iteration  % 迭代直到达到最大代数
        leng = size(ERP, 1);  % 当前仓库中的个体数
        r2 = rand;  % 随机数
        
        % 更新前20%的个体
        for i = 1 : pNum1
            if (r2 < 0.9)
                r1 = rand;  % 随机数
                a = rand;  % 随机数
                if (a > 0.1)
                    a = 1;
                else
                    a = -1;
                end
                worse = ERP(randperm(leng, 1), 1:D);  % 随机选择一个较差的个体
                % 更新位置，基于当前个体与较差个体的差异
                newPOS(i, 1:D) = POS(i, 1:D) + 0.3 * abs(POS(i, 1:D) - worse) + a * 0.1 * POS_ad(i, 1:D); % 方程 (1)
            else
                aaa = randperm(180, 1);  % 随机选择一个角度
                if (aaa == 0 || aaa == 90 || aaa == 180)
                    newPOS(i, 1:D) = POS(i, 1:D);  % 不更新
                end
                theta = aaa * pi / 180;  % 转换为弧度
                % 更新位置，基于角度的变化
                newPOS(i, 1:D) = POS(i, 1:D) + tan(theta) .* abs(POS(i, 1:D) - POS_ad(i, 1:D));    % 方程 (2)
            end
        end
        
        % 计算收敛因子R
        R = 1 - Iteration / Max_iteration;
        
        % 选择最佳个体用于更新位置
        bestXX = ERP(randperm(leng, 1), 1:D);
        
        % 更新位置，根据收敛因子R
        % 方程 (3)
        Xnew1 = bestXX .* (1 - R);
        Xnew2 = bestXX .* (1 + R);
        Xnew1 = bound(Xnew1, UB, LB);  % 越界判断
        Xnew2 = bound(Xnew2, UB, LB);  % 越界判断
        
        bestX = ERP(randperm(leng, 1), 1:D);
        % 方程 (5)
        Xnew11 = bestX .* (1 - R);
        Xnew22 = bestX .* (1 + R);
        Xnew11 = bound(Xnew11, UB, LB);  % 越界判断
        Xnew22 = bound(Xnew22, UB, LB);  % 越界判断
        
        % 更新中间40%的个体
        for i = (pNum1 + 1) : pNum2  % 方程 (4)
            newPOS(i, 1:D) = bestXX + (rand(1, D) .* (POS(i, 1:D) - Xnew1) + rand(1, D) .* (POS(i, 1:D) - Xnew2));
        end
        
        % 更新中间23%的个体
        for i = pNum2 + 1 : pNum3  % 方程 (6)
            newPOS(i, 1:D) = POS(i, 1:D) + (randn(1) .* (POS(i, 1:D) - Xnew11) + (rand(1, D) .* (POS(i, 1:D) - Xnew22)));
        end
        
        % 更新剩余个体
        for j = pNum3 + 1 : SearchAgents_no  % 方程 (7)
            newPOS(j, 1:D) = bestX + randn(1, D) .* ((abs(POS(j, 1:D) - bestXX)) + (abs(POS(j, 1:D) - bestX))) / 2;
        end
        
        %% 计算新位置的目标函数值
        for i = 1 : SearchAgents_no
            newPOS(i, 1:D) = bound(newPOS(i, 1:D), UB, LB);  % 越界判断
            newPOS(i, D + 1 : K) = evaluate_objective(newPOS(i, 1:D));  % 计算目标函数值
            
            % 判断是否更新适应度存储
            dom_less = 0;
            dom_equal = 0;
            dom_more = 0;
            for k = 1 : M
                if (newPOS(i, D + k) < POS(i, D + k))
                    dom_less = dom_less + 1;
                elseif (newPOS(i, D + k) == POS(i, D + k))
                    dom_equal = dom_equal + 1;
                else
                    dom_more = dom_more + 1;
                end
            end
            
            if dom_more == 0 && dom_equal ~= M
                % 如果新个体在所有目标上不劣于当前个体，更新位置
                POS_ad(i, 1:K) = POS(i, 1:K);
                POS(i, 1:K) = newPOS(i, 1:K);
            else
                % 否则，更新适应度存储
                POS_ad(i, 1:K) = newPOS(i, 1:K);
            end
        end
        
        %% 非支配排序和仓库更新
        pos_com = [POS(:, 1:K) ; POS_ad];  % 合并当前位置和适应度存储
        intermediate_pos = non_domination_sort_mod(pos_com, M, D);  % 对合并后的种群进行非支配排序
        POS = replace_chromosome(intermediate_pos, M, D, Nr);  % 替换仓库中的个体
        
        DOMINATED = checkDomination(POS(:, D + 1 : D + M));  % 检查支配关系
        Pop = POS(~DOMINATED, :);  % 移除被支配的个体
        ERP = Pop(:, 1:K + 1);  % 更新仓库中的个体
        
        %% 显示迭代信息
        if rem(Iteration, ishow) == 0
            disp(['NSDBO Iteration ' num2str(Iteration) ': Number of solutions in the archive = ' num2str(size(ERP, 1))]);
        end
        Iteration = Iteration + 1;  % 增加迭代次数
        
        %% 绘制Pareto前沿
        h_fig = figure(1);  % 创建或选择图形窗口
        
        if (numOfObj == 2)
            pl_data = ERP(:, D + 1 : D + M);  % 提取用于绘图的数据
            POS_fit = sortrows(pl_data, 2);  % 按第二个目标函数排序
            figure(h_fig); 
            try delete(h_par); end  % 删除之前的点
            h_par = plot(POS_fit(:, 1), POS_fit(:, 2), 'or'); hold on;  % 绘制Pareto前沿点
            if (isfield(MultiObj, 'truePF'))
                try delete(h_pf); end  % 删除之前的真实Pareto前沿
                h_pf = plot(MultiObj.truePF(:, 1), MultiObj.truePF(:, 2), '.k'); hold on;  % 绘制真实Pareto前沿
            end
            title(name);  % 设置标题
            xlabel('f1'); ylabel('f2');  % 设置坐标轴标签
            drawnow;  % 更新图形
        end
        
        if (numOfObj == 3)
            pl_data = ERP(:, D + 1 : D + M);  % 提取用于绘图的数据
            POS_fit = sortrows(pl_data, 3);  % 按第三个目标函数排序
            figure(h_fig); 
            try delete(h_par); end  % 删除之前的点
            h_par = plot3(POS_fit(:, 1), POS_fit(:, 2), POS_fit(:, 3), 'or'); hold on;  % 绘制Pareto前沿点
            if (isfield(MultiObj, 'truePF'))
                try delete(h_pf); end  % 删除之前的真实Pareto前沿
                h_pf = plot3(MultiObj.truePF(:, 1), MultiObj.truePF(:, 2), MultiObj.truePF(:, 3), '.k'); hold on;  % 绘制真实Pareto前沿
            end
            title(name);  % 设置标题
            grid on;  % 显示网格
            xlabel('f1'); ylabel('f2'); zlabel('f3');  % 设置坐标轴标签
            drawnow;  % 更新图形
        end
    end
    toc  % 结束计时
    
    % 返回最终仓库中的个体
    f = ERP;
    hold off;
    
    % 添加图例
    if (isfield(MultiObj, 'truePF'))
        legend('NSDBO', 'TruePF');
    else
        legend('NSDBO');
    end
end

%% 辅助函数

% bound 函数
% 检查并修正个体的决策变量是否超出边界
% 输入:
%   a  - 决策变量向量
%   ub - 上界向量
%   lb - 下界向量
% 输出:
%   a  - 修正后的决策变量向量
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);  % 将超过上界的值设为上界
    a(a < lb) = lb(a < lb);  % 将低于下界的值设为下界
    % 以下两行代码用于在越界时重新随机生成值
    % a(a > ub) = rand(1).*(ub(a > ub) - lb(a > ub)) + lb(a > ub);
    % a(a < lb) = rand(1).*(ub(a < lb) - lb(a < lb)) + lb(a < lb);
end

% dominates 函数
% 判断个体x是否支配个体y
% 输入:
%   x - 个体x的目标函数值矩阵
%   y - 个体y的目标函数值矩阵
% 输出:
%   d - 支配结果向量，1表示x支配y，0表示不支配
function d = dominates(x, y)
    d = all(x <= y, 2) & any(x < y, 2);  % x在所有目标上不劣于y，且至少在一个目标上优于y
end

% checkDomination 函数
% 检查种群中个体的支配关系
% 输入:
%   fitness - 种群中个体的目标函数值矩阵
% 输出:
%   dom_vector - 支配向量，1表示被支配，0表示不被支配
function dom_vector = checkDomination(fitness)
    Np = size(fitness, 1);  % 种群规模
    dom_vector = zeros(Np, 1);  % 初始化支配向量
    
    % 生成所有可能的个体对排列
    all_perm = nchoosek(1:Np, 2);  % 生成所有2个个体的组合
    all_perm = [all_perm; [all_perm(:, 2) all_perm(:, 1)]];  % 添加反向排列
    
    % 判断哪些个体对满足x支配y
    d = dominates(fitness(all_perm(:, 1), :), fitness(all_perm(:, 2), :));
    
    % 找出被支配的个体索引
    dominated_particles = unique(all_perm(d == 1, 2));
    dom_vector(dominated_particles) = 1;  % 标记被支配的个体
end
