function [Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)
% 处理存档中的解，当存档已满时，使用轮盘赌选择算法删除一些解，保持存档的最大大小
% 输入:
%   Archive_X: 存档中的解的决策变量矩阵
%   Archive_F: 存档中的解的目标函数值矩阵
%   Archive_member_no: 存档中的解的数量
%   Archive_mem_ranks: 存档中每个解的排序值
%   ArchiveMaxSize: 存档的最大容量
% 输出:
%   Archive_X_Chopped: 更新后的存档决策变量矩阵
%   Archive_F_Chopped: 更新后的存档目标函数值矩阵
%   Archive_mem_ranks_updated: 更新后的排序值矩阵
%   Archive_member_no: 更新后的存档解的数量

% 如果存档大小超过最大容量，则删除一些解
for i = 1:size(Archive_F, 1) - ArchiveMaxSize
    % 使用轮盘赌选择算法选择一个解
    index = RouletteWheelSelection(Archive_mem_ranks);
    
    % 删除选择的解
    Archive_X = [Archive_X(1:index-1, :); Archive_X(index+1:Archive_member_no, :)];
    Archive_F = [Archive_F(1:index-1, :); Archive_F(index+1:Archive_member_no, :)];
    Archive_mem_ranks = [Archive_mem_ranks(1:index-1), Archive_mem_ranks(index+1:Archive_member_no)];
    
    % 更新存档解的数量
    Archive_member_no = Archive_member_no - 1;
end

% 返回更新后的存档数据
Archive_X_Chopped = Archive_X;
Archive_F_Chopped = Archive_F;
Archive_mem_ranks_updated = Archive_mem_ranks;
