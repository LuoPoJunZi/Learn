% -------------------------------------------------------------
% 文件名: UpdateArchive.m
% 功能: 更新存档（Archive），将新产生的个体加入存档，并保持存档中的非支配解
%       通过判断支配关系，移除被支配的个体，保留非支配的个体
% 输入:
%       Archive_X        - 当前存档中个体的决策变量矩阵（每行一个个体）
%       Archive_F        - 当前存档中个体的目标函数值矩阵（每行一个个体）
%       Particles_X      - 新产生的个体的决策变量矩阵（每行一个个体）
%       Particles_F      - 新产生的个体的目标函数值矩阵（每行一个个体）
%       Archive_member_no - 当前存档中的个体数量
% 输出:
%       Archive_X_updated    - 更新后的存档中个体的决策变量矩阵
%       Archive_F_updated    - 更新后的存档中个体的目标函数值矩阵
%       Archive_member_no    - 更新后的存档中个体数量
% 作者: [您的名字]
% 日期: [日期]
% -------------------------------------------------------------

function [Archive_X_updated, Archive_F_updated, Archive_member_no] = UpdateArchive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no)
    % 将新个体加入存档
    Archive_X_temp = [Archive_X; Particles_X'];  % 假设 Particles_X 的每行是一个个体，转置后与 Archive_X 纵向拼接
    Archive_F_temp = [Archive_F; Particles_F];   % 将新个体的目标函数值纵向拼接到存档中
    
    % 初始化一个标记数组，用于标记哪些个体被支配（1表示被支配，0表示非支配）
    o = zeros(1, size(Archive_F_temp, 1));
    
    % 遍历存档中的每个个体，判断其是否被其他个体支配
    for i = 1:size(Archive_F_temp, 1)
        o(i) = 0;  % 初始化当前个体的支配标记为0（非支配）
        
        % 与存档中的其他个体进行比较
        for j = 1:i-1
            if any(Archive_F_temp(i, :) ~= Archive_F_temp(j, :))
                if dominates(Archive_F_temp(i, :), Archive_F_temp(j, :))
                    o(j) = 1;  % 如果第i个个体支配第j个个体，则标记第j个个体为被支配
                elseif dominates(Archive_F_temp(j, :), Archive_F_temp(i, :))
                    o(i) = 1;  % 如果第j个个体支配第i个个体，则标记第i个个体为被支配
                    break;      % 一旦被支配，跳出内层循环
                end
            else
                % 如果两个个体的目标函数值完全相同，则都标记为被支配
                o(j) = 1;
                o(i) = 1;
            end
        end
    end
    
    % 初始化更新后的存档矩阵
    Archive_member_no = 0;      % 重置存档中的个体数量
    Archive_X_updated = [];      % 初始化更新后的决策变量矩阵
    Archive_F_updated = [];      % 初始化更新后的目标函数值矩阵
    index = 0;                    % 初始化索引变量（未使用）
    
    % 遍历所有临时存档中的个体，将非支配的个体加入更新后的存档
    for i = 1:size(Archive_X_temp, 1)
        if o(i) == 0
            Archive_member_no = Archive_member_no + 1;  % 增加存档中个体的数量
            Archive_X_updated(Archive_member_no, :) = Archive_X_temp(i, :);  % 添加决策变量
            Archive_F_updated(Archive_member_no, :) = Archive_F_temp(i, :);  % 添加目标函数值
        else
            index = index + 1;  % 增加被支配个体的计数（可用于调试或分析）
            % 被支配的个体可以存储在其他变量中，当前代码中被注释掉
            % dominated_X(index, :) = Archive_X_temp(i, :);
            % dominated_F(index, :) = Archive_F_temp(i, :);
        end
    end
end
