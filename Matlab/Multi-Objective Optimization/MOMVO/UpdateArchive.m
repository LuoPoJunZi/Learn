function [Archive_X_updated, Archive_F_updated, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no)
% 更新非支配解归档，添加新解并去除被支配的解
% 输入：
%   Archive_X - 当前归档集合中的解的决策变量
%   Archive_F - 当前归档集合中解的目标值
%   Particles_X - 新解的决策变量
%   Particles_F - 新解的目标值
%   Archive_member_no - 当前归档集合中的解的数量
% 输出：
%   Archive_X_updated - 更新后的归档集合中的解的决策变量
%   Archive_F_updated - 更新后的归档集合中的解的目标值
%   Archive_member_no - 更新后的归档集合中的解的数量

% 将归档集合和新解集合合并，形成一个临时集合
Archive_X_temp = [Archive_X ; Particles_X];
Archive_F_temp = [Archive_F ; Particles_F];

% 初始化一个标记数组 o，用于标记解是否被支配
o = zeros(1, size(Archive_F_temp, 1));

% 遍历每个解，检查是否有支配关系
for i = 1:size(Archive_F_temp, 1)
    o(i) = 0; % 假设当前解 i 不被支配
    for j = 1:i-1
        % 如果解 i 和解 j 不相同
        if any(Archive_F_temp(i, :) ~= Archive_F_temp(j, :))
            % 检查是否解 i 支配解 j
            if dominates(Archive_F_temp(i, :), Archive_F_temp(j, :))
                o(j) = 1; % 解 j 被解 i 支配，标记 j 为 1
            % 检查是否解 j 支配解 i
            elseif dominates(Archive_F_temp(j, :), Archive_F_temp(i, :))
                o(i) = 1; % 解 i 被解 j 支配，标记 i 为 1
                break; % 退出循环，因为解 i 已经被支配
            end
        else
            % 解 i 和解 j 相同，标记为已支配
            o(j) = 1;
            o(i) = 1;
        end
    end
end

% 重新初始化归档集合
Archive_member_no = 0;
index = 0;
for i = 1:size(Archive_X_temp, 1)
    % 仅保留未被支配的解
    if o(i) == 0
        Archive_member_no = Archive_member_no + 1; % 更新解的数量
        Archive_X_updated(Archive_member_no, :) = Archive_X_temp(i, :);
        Archive_F_updated(Archive_member_no, :) = Archive_F_temp(i, :);
    else
        index = index + 1;
    end
end

end
