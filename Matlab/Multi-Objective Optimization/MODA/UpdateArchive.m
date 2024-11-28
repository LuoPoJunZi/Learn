% UpdateArchive.m - 更新存档中的非支配解
%
% 该函数将当前种群中的解添加到存档中，并移除被支配的解，确保存档中只包含非支配解。
%
% 输入:
%   Archive_X - 存档中所有解的决策变量矩阵
%   Archive_F - 存档中所有解的目标函数值矩阵
%   Particles_X - 当前种群中所有解的决策变量矩阵
%   Particles_F - 当前种群中所有解的目标函数值矩阵
%   Archive_member_no - 存档中当前成员的数量
%
% 输出:
%   Archive_X_updated - 更新后的存档中解的决策变量矩阵
%   Archive_F_updated - 更新后的存档中解的目标函数值矩阵
%   Archive_member_no - 更新后的存档中成员数量

function [Archive_X_updated, Archive_F_updated, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no)
    % 将当前种群中的解添加到临时存档中
    Archive_X_temp = [Archive_X ; Particles_X'];
    Archive_F_temp = [Archive_F ; Particles_F];
    
    % 初始化一个标志数组，用于标记被支配的解
    o = zeros(1, size(Archive_F_temp, 1));
    
    % 遍历临时存档中的每个解，检查是否被其他解支配
    for i = 1:size(Archive_F_temp, 1)
        o(i) = 0; % 初始化当前解的支配标志为 0（未被支配）
        for j = 1:i-1
            if any(Archive_F_temp(i, :) ~= Archive_F_temp(j, :))
                if dominates(Archive_F_temp(i, :), Archive_F_temp(j, :))
                    o(j) = 1; % 如果解 i 支配解 j，则标记解 j 被支配
                elseif dominates(Archive_F_temp(j, :), Archive_F_temp(i, :))
                    o(i) = 1; % 如果解 j 支配解 i，则标记解 i 被支配
                    break;     % 解 i 被支配，跳出内层循环
                end
            else
                % 如果解 i 和解 j 在所有目标上相等，则都被标记为被支配
                o(j) = 1;
                o(i) = 1;
            end
        end
    end
    
    % 初始化更新后的存档变量
    Archive_member_no = 0;
    index = 0;
    
    % 遍历临时存档，保留未被支配的解
    for i = 1:size(Archive_X_temp, 1)
        if o(i) == 0
            Archive_member_no = Archive_member_no + 1;
            Archive_X_updated(Archive_member_no, :) = Archive_X_temp(i, :);
            Archive_F_updated(Archive_member_no, :) = Archive_F_temp(i, :);
        else
            index = index + 1; % 可选：记录被移除的解数量
        end
    end
end
