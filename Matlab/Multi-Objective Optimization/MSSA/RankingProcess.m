% -------------------------------------------------------------
% 文件名: RankingProcess.m
% 功能: 对存档中的个体进行排名处理，基于各个个体的目标函数值
%       通过计算每个个体在邻域中的数量来确定其排名，邻域的定义基于距离阈值
% 输入:
%       Archive_F     - 存档中个体的目标函数值矩阵（每行一个个体）
%       ArchiveMaxSize - 存档的最大容量
%       obj_no        - 目标函数的数量
% 输出:
%       ranks         - 存档中每个个体的排名（数组）
% 作者: [您的名字]
% 日期: [日期]
% -------------------------------------------------------------

function ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no)
    % 使用全局变量来存储最小值和最大值，用于计算距离阈值
    global my_min;
    global my_max;
    
    % 如果存档中只有一个个体，初始化my_min和my_max为该个体的目标函数值
    if size(Archive_F, 1) == 1 && size(Archive_F, 2) == obj_no
        my_min = Archive_F;
        my_max = Archive_F;
    else
        % 计算存档中每个目标的最小值
        my_min = min(Archive_F);
        % 计算存档中每个目标的最大值
        my_max = max(Archive_F);
    end
    
    % 计算距离阈值r，基于目标函数的范围划分为20个区间
    r = (my_max - my_min) / 20;
    
    % 初始化排名数组，长度为存档中个体的数量
    ranks = zeros(1, size(Archive_F, 1));
    
    % 对存档中的每个个体进行排名计算
    for i = 1:size(Archive_F, 1)
        ranks(i) = 0;  % 初始化当前个体的排名
        
        % 与存档中的其他个体进行比较
        for j = 1:size(Archive_F, 1)
            flag = 0;  % 标志变量，检查当前个体是否在所有维度上与比较个体邻近
            
            % 对每个目标函数进行检查
            for k = 1:obj_no
                % 如果在第k个目标上，两个个体的差值小于阈值r(k)，则认为在该维度上是邻近的
                if (abs(Archive_F(j, k) - Archive_F(i, k)) < r(k))
                    flag = flag + 1;
                end
            end
            
            % 如果在所有目标函数上都是邻近的，则增加当前个体的排名
            if flag == obj_no
                ranks(i) = ranks(i) + 1;
            end
        end
    end
end
