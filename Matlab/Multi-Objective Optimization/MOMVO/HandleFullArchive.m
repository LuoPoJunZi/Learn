function [Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)
% HandleFullArchive 函数用于在归档满时通过删除低优先级个体来管理归档大小
% 输入：
%   Archive_X - 归档中个体的决策变量矩阵
%   Archive_F - 归档中个体的目标函数值矩阵
%   Archive_member_no - 当前归档中的个体数量
%   Archive_mem_ranks - 归档中个体的排名，用于选择要删除的个体
%   ArchiveMaxSize - 归档的最大容量
% 输出：
%   Archive_X_Chopped - 调整后的归档决策变量矩阵
%   Archive_F_Chopped - 调整后的归档目标函数值矩阵
%   Archive_mem_ranks_updated - 调整后的归档个体排名
%   Archive_member_no - 调整后的归档中个体数量

% 当归档中的个体数量超过最大容量时，逐步删除低优先级个体
for i = 1:size(Archive_F,1) - ArchiveMaxSize
    % 使用轮盘赌法选择要删除的个体索引
    index = RouletteWheelSelection(Archive_mem_ranks);
    
    % 从决策变量矩阵中删除选定的个体
    Archive_X = [Archive_X(1:index-1, :); Archive_X(index+1:Archive_member_no, :)];
    % 从目标函数值矩阵中删除选定的个体
    Archive_F = [Archive_F(1:index-1, :); Archive_F(index+1:Archive_member_no, :)];
    % 从排名列表中删除选定的个体排名
    Archive_mem_ranks = [Archive_mem_ranks(1:index-1), Archive_mem_ranks(index+1:Archive_member_no)];
    
    % 更新归档个体数量
    Archive_member_no = Archive_member_no - 1;
end

% 输出处理后的归档信息
Archive_X_Chopped = Archive_X;
Archive_F_Chopped = Archive_F;
Archive_mem_ranks_updated = Archive_mem_ranks;

end
