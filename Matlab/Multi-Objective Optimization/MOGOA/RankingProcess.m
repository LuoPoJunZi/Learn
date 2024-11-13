function ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no)
% 计算存档中解的排名，用于非支配排序
% 输入:
%   Archive_F: 存档中的解的目标函数值矩阵，每一行对应一个解的目标值
%   ArchiveMaxSize: 存档的最大容量（虽然此函数中未直接使用）
%   obj_no: 目标函数的数量
% 输出:
%   ranks: 存档中每个解的排名，排名越小表示解越优

global my_min;  % 全局最小值
global my_max;  % 全局最大值

% 如果存档只有一个解，直接设置最小值和最大值为该解的目标值
if size(Archive_F, 1) == 1 && size(Archive_F, 2) == 2
    my_min = Archive_F;
    my_max = Archive_F;
else
    % 计算存档目标函数值的最小值和最大值
    my_min = min(Archive_F);
    my_max = max(Archive_F);
end

% 设置一个区间值，用于在计算解的邻域时使用
r = (my_max - my_min) / 20;

% 初始化排名数组
ranks = zeros(1, size(Archive_F, 1));

% 遍历每一个解，计算其在存档中的排名
for i = 1:size(Archive_F, 1)
    ranks(i) = 0;  % 初始化排名为0
    for j = 1:size(Archive_F, 1)
        flag = 0;  % 用于判断当前解是否与其他解在所有目标维度上都在邻域内
        
        for k = 1:obj_no
            % 判断解 j 和解 i 在目标函数值上是否相差很小（即是否在邻域内）
            if abs(Archive_F(j, k) - Archive_F(i, k)) < r(k)
                flag = flag + 1;  % 如果相差较小，则认为这两个解在第 k 维上属于邻域
            end
        end
        
        % 如果解 j 和解 i 在所有目标维度上都属于邻域，则给解 i 增加排名
        if flag == obj_no
            ranks(i) = ranks(i) + 1;
        end
    end
end
end
