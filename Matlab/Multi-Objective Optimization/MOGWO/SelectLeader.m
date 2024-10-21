function rep_h = SelectLeader(rep, beta)
    % 选择领导者函数
    % 输入:
    %   rep - 存储个体的结构体数组
    %   beta - 选择压力参数（默认值为1）
    % 输出:
    %   rep_h - 选择的领导者个体

    if nargin < 2
        beta = 1;  % 如果未提供beta参数，默认为1
    end

    % 获取占用的单元格及其成员数量
    [occ_cell_index, occ_cell_member_count] = GetOccupiedCells(rep);
    
    % 计算每个单元格的选择概率
    p = occ_cell_member_count.^(-beta);  % 概率与成员数量的负指数成反比
    p = p / sum(p);  % 归一化概率，使其总和为1
    
    % 使用轮盘赌选择法选择一个单元格
    selected_cell_index = occ_cell_index(RouletteWheelSelection(p));
    
    % 获取被选中单元格的所有成员
    GridIndices = [rep.GridIndex];
    selected_cell_members = find(GridIndices == selected_cell_index);
    
    % 随机选择单元格中的一个个体
    n = numel(selected_cell_members);  % 成员数量
    selected_memebr_index = randi([1, n]);  % 随机选择一个索引
    
    h = selected_cell_members(selected_memebr_index);  % 获取选中的个体索引
    
    rep_h = rep(h);  % 返回选择的领导者个体
end
