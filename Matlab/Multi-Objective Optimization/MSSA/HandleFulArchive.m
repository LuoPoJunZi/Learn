% -------------------------------------------------------------
% 文件名: HandleFulArchive.m
% 功能: 处理当存档（Archive）满时，通过轮盘赌选择策略移除部分个体
%       该函数根据个体的等级（rank）来进行选择，优先保留等级较高的个体
% 输入:
%       Archive_X          - 存档中个体的决策变量矩阵（每行一个个体）
%       Archive_F          - 存档中个体的目标函数值矩阵（每行一个个体）
%       Archive_member_no  - 当前存档中的个体数量
%       Archive_mem_ranks  - 存档中每个个体的等级（数组）
%       ArchiveMaxSize     - 存档的最大容量
% 输出:
%       Archive_X_Chopped       - 处理后的存档个体决策变量矩阵
%       Archive_F_Chopped       - 处理后的存档个体目标函数值矩阵
%       Archive_mem_ranks_updated - 更新后的存档个体等级数组
%       Archive_member_no       - 更新后的存档中个体数量
% 作者: [您的名字]
% 日期: [日期]
% -------------------------------------------------------------

function [Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no] = HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)
    % 循环删除多余的个体，直到存档大小不超过最大容量
    for i = 1:(size(Archive_F, 1) - ArchiveMaxSize)
        % 使用轮盘赌选择方法根据个体等级选择要移除的个体的索引
        index = RouletteWheelSelection(Archive_mem_ranks);
        
        % 从存档中移除选中的个体
        % 更新决策变量矩阵
        Archive_X = [Archive_X(1:index-1, :); Archive_X(index+1:Archive_member_no, :)];
        
        % 更新目标函数值矩阵
        Archive_F = [Archive_F(1:index-1, :); Archive_F(index+1:Archive_member_no, :)];
        
        % 更新等级数组
        Archive_mem_ranks = [Archive_mem_ranks(1:index-1) Archive_mem_ranks(index+1:Archive_member_no)];
        
        % 更新存档中个体的数量
        Archive_member_no = Archive_member_no - 1;
    end
    
    % 将处理后的存档数据赋值给输出变量
    Archive_X_Chopped = Archive_X;
    Archive_F_Chopped = Archive_F;
    Archive_mem_ranks_updated = Archive_mem_ranks;
end
