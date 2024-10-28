function ranks = RankingProcess(Archive_F, ArchiveMaxSize, obj_no)
% 计算归档集合中每个解的拥挤度等级，用于非支配排序中的拥挤度排序
% 输入：
%   Archive_F - 归档集合中每个解的目标值矩阵
%   ArchiveMaxSize - 归档集合的最大容量
%   obj_no - 目标函数的数量
% 输出：
%   ranks - 每个解的拥挤度等级（等级越高表示周围解的密度越大）

% 计算每个目标的最小值和最大值
my_min = min(Archive_F);
my_max = max(Archive_F);

% 如果归档集合中只有一个解，最小值和最大值相等
if size(Archive_F, 1) == 1
    my_min = Archive_F;
    my_max = Archive_F;
end

% 将每个目标的范围划分为20个区间，用于计算邻域
r = (my_max - my_min) / 20;

% 初始化拥挤度等级数组
ranks = zeros(1, size(Archive_F, 1));

% 遍历归档集合中的每个解
for i = 1:size(Archive_F, 1)
    ranks(i) = 0; % 初始化解 i 的拥挤度等级

    % 比较解 i 与其他解的邻域关系
    for j = 1:size(Archive_F, 1)
        flag = 0; % 标记当前解 j 是否在解 i 的邻域内

        % 检查解 j 是否在解 i 的所有目标维度上的邻域内
        for k = 1:obj_no
            if abs(Archive_F(j, k) - Archive_F(i, k)) < r(k)
                flag = flag + 1;
            end
        end

        % 如果解 j 在解 i 的所有目标维度上都在邻域内，增加解 i 的拥挤度等级
        if flag == obj_no
            ranks(i) = ranks(i) + 1;
        end
    end
end
end
