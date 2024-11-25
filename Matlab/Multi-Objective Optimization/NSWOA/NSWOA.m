function f = NSWOA(D, M, LB, UB, Pop, SearchAgents_no, Max_iteration, ishow)
%% 非支配鲸鱼优化算法（NSWOA）
% NSWOA 由 Pradeep Jangir 开发
% 输入参数：
% f - 最优适应度值
% X - 最优解
% D - 问题维度
% M - 目标函数个数
% LB - 下界约束
% UB - 上界约束
% Pop - 初始种群
% SearchAgents_no - 搜索代理（个体）数量
% Max_iteration - 最大迭代次数
% ishow - 显示进度的间隔

%% 算法变量初始化
K = D + M; % 解向量和目标值的总维度
Whale_pos = Pop(:, 1:K+1); % 存储种群位置和非支配等级
Whale_pos_ad = zeros(SearchAgents_no, K); % 存储进化后的种群

%% 优化主循环
Iteration = 1;
while Iteration <= Max_iteration
    % 对于种群中的每个个体
    for i = 1:SearchAgents_no
        % 随机选择另一个个体进行比较
        j = floor(rand() * SearchAgents_no) + 1;
        while j == i
            j = floor(rand() * SearchAgents_no) + 1;
        end
        
        % 缩放因子，用于覆盖最佳解
        SF = round(1 + rand);
        
        % 从第一非支配前沿随机选择一个解
        Ffront = Whale_pos((find(Whale_pos(:, K+1) == 1)), :);
        ri = floor(size(Ffront, 1) * rand()) + 1;
        sorted_population = Ffront(ri, 1:D);
        
        % 计算新解
        Whale_posNew1 = Whale_pos(i, 1:D) + rand(1, D) .* (sorted_population - SF .* Whale_pos(i, 1:D));
        
        % 处理边界条件
        Whale_posNew1 = bound(Whale_posNew1(:, 1:D), UB, LB);
        
        % 计算目标函数值
        Whale_posNew1(:, D + 1:K) = evaluate_objective(Whale_posNew1(:, 1:D));
        
        % 非支配性检查
        dom_less = 0;
        dom_equal = 0;
        dom_more = 0;
        for k = 1:M
            if (Whale_posNew1(:, D+k) < Whale_pos(i, D+k))
                dom_less = dom_less + 1;
            elseif (Whale_posNew1(:, D+k) == Whale_pos(i, D+k))
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        
        % 如果新解支配当前解
        if dom_more == 0 && dom_equal ~= M
            Whale_pos_ad(i, 1:K) = Whale_pos(i, 1:K);
            Whale_pos(i, 1:K) = Whale_posNew1(:, 1:K);
        else
            Whale_pos_ad(i, 1:K) = Whale_posNew1;
        end
    end
    
    % 输出当前代信息
    if rem(Iteration, ishow) == 0
        fprintf('Generation: %d\n', Iteration);
    end
    
    % 合并当前种群与进化种群
    Whale_pos_com = [Whale_pos(:, 1:K); Whale_pos_ad];
    
    % 非支配排序
    intermediate_Whale_pos = non_domination_sort_mod(Whale_pos_com, M, D);
    
    % 替换种群
    Pop = replace_chromosome(intermediate_Whale_pos, M, D, SearchAgents_no);
    Whale_pos = Pop(:, 1:K+1);
    Iteration = Iteration + 1;
end

% 返回最终种群
f = Whale_pos;

%% 边界处理函数
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub); % 超出上界修正
    a(a < lb) = lb(a < lb); % 超出下界修正
end
