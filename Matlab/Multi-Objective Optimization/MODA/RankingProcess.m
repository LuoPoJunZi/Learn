% RankingProcess.m - 存档中解的排名处理
%
% 该函数用于对存档中的解进行排名，以便在选择食物和敌人时考虑解的分布和多样性。
%
% 输入:
%   Archive_F - 存档中所有解的目标函数值矩阵（每行代表一个解）
%   ArchiveMaxSize - 存档的最大容量
%   obj_no - 目标函数的数量
%
% 输出:
%   ranks - 每个存档解的排名（越高表示解位于人群较少的区域）

function ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no)
    % 计算每个目标的最小值和最大值
    my_min = min(Archive_F);
    my_max = max(Archive_F);
    
    % 如果存档中只有一个解，最小值和最大值都为该解的目标值
    if size(Archive_F, 1) == 1
        my_min = Archive_F;
        my_max = Archive_F;
    end
    
    % 计算每个目标的分辨率 r，用于确定邻域
    r = (my_max - my_min) / 20;
    
    % 初始化排名数组
    ranks = zeros(1, size(Archive_F, 1));
    
    % 对每个解进行排名
    for i = 1:size(Archive_F, 1)
        ranks(i) = 0;  % 初始化当前解的排名
        for j = 1:size(Archive_F, 1)
            flag = 0; % 标志变量，判断 j 解是否在 i 解的邻域内
            for k = 1:obj_no
                % 如果 j 解在第 k 个目标上与 i 解的差小于 r(k)
                if (abs(Archive_F(j, k) - Archive_F(i, k)) < r(k))
                    flag = flag + 1;
                end
            end
            % 如果 j 解在所有目标上都在 i 解的邻域内，则增加 i 解的排名
            if flag == obj_no
                ranks(i) = ranks(i) + 1;
            end
        end
    end
end
