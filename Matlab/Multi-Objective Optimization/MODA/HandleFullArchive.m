% HandleFullArchive.m - 处理存档已满的情况，移除部分存档成员
%
% 输入:
%   Archive_X - 存档中所有解的决策变量
%   Archive_F - 存档中所有解的目标函数值
%   Archive_member_no - 存档中成员的数量
%   Archive_mem_ranks - 存档中各成员的等级
%   ArchiveMaxSize - 存档的最大容量
%
% 输出:
%   Archive_X_Chopped - 处理后的存档决策变量
%   Archive_F_Chopped - 处理后的存档目标函数值
%   Archive_mem_ranks_updated - 更新后的存档成员等级
%   Archive_member_no - 更新后的存档成员数量

function [Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)
    % 当存档成员数量超过最大容量时，循环移除多余的成员
    for i = 1:(size(Archive_F, 1) - ArchiveMaxSize)
        % 通过轮盘赌选择要移除的成员索引
        index = RouletteWheelSelection(Archive_mem_ranks);
        
        % 移除选定的成员的决策变量
        Archive_X = [Archive_X(1:index-1, :) ; Archive_X(index+1:Archive_member_no, :)];
        
        % 移除选定的成员的目标函数值
        Archive_F = [Archive_F(1:index-1, :) ; Archive_F(index+1:Archive_member_no, :)];
        
        % 移除选定的成员的等级
        Archive_mem_ranks = [Archive_mem_ranks(1:index-1) Archive_mem_ranks(index+1:Archive_member_no)];
        
        % 更新存档成员数量
        Archive_member_no = Archive_member_no - 1;
    end
    
    % 输出处理后的存档数据
    Archive_X_Chopped = Archive_X;
    Archive_F_Chopped = Archive_F;
    Archive_mem_ranks_updated = Archive_mem_ranks;
end
