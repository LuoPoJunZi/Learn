function [Archive_X_updated, Archive_F_updated, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no)
% 更新存档，将新的粒子解与现有解进行比较，保留非支配解
% 输入:
%   Archive_X: 当前存档的解的决策变量矩阵
%   Archive_F: 当前存档的解的目标函数值矩阵
%   Particles_X: 新的粒子解的决策变量矩阵
%   Particles_F: 新的粒子解的目标函数值矩阵
%   Archive_member_no: 当前存档中解的数量
% 输出:
%   Archive_X_updated: 更新后的存档解的决策变量矩阵
%   Archive_F_updated: 更新后的存档解的目标函数值矩阵
%   Archive_member_no: 更新后的存档中解的数量

% 将新的粒子解加入存档临时数组
Archive_X_temp = [Archive_X; Particles_X'];
Archive_F_temp = [Archive_F; Particles_F];

% 初始化一个标志数组，表示每个解是否被支配
o = zeros(1, size(Archive_F_temp, 1));

% 遍历所有解，进行非支配排序
for i = 1:size(Archive_F_temp, 1)
    o(i) = 0;  % 初始化为未被支配
    for j = 1:i-1
        % 如果两个解的目标函数值不同，进行支配关系的判断
        if any(Archive_F_temp(i, :) ~= Archive_F_temp(j, :))
            if dominates(Archive_F_temp(i, :), Archive_F_temp(j, :))
                o(j) = 1;  % 解 j 被解 i 支配
            elseif dominates(Archive_F_temp(j, :), Archive_F_temp(i, :))
                o(i) = 1;  % 解 i 被解 j 支配
                break;
            end
        else
            o(j) = 1;  % 如果目标函数值相同，认为两者互不支配
            o(i) = 1;
        end
    end
end

% 更新存档的解
Archive_member_no = 0;  % 初始化存档中解的数量
index = 0;  % 用于记录被支配的解的数量
for i = 1:size(Archive_X_temp, 1)
    if o(i) == 0  % 如果该解不被支配，则加入存档
        Archive_member_no = Archive_member_no + 1;
        Archive_X_updated(Archive_member_no, :) = Archive_X_temp(i, :);
        Archive_F_updated(Archive_member_no, :) = Archive_F_temp(i, :);
    else
        index = index + 1;  % 如果该解被支配，则不加入存档
    end
end
end
