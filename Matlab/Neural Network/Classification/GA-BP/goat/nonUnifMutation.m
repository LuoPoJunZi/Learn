function parent = nonUnifMutation(parent, bounds, Ops)
% nonUnifMutation - 非均匀突变函数，根据非均匀概率分布改变父代的参数
%
% 输入参数:
%   parent - 父代个体的参数向量
%   bounds - 变量的边界矩阵，每行表示一个变量的 [高界 低界]
%   Ops    - 选项向量 [当前代数 突变次数 最大代数 b]
%
% 输出参数:
%   parent - 突变后的子代个体参数向量

    %% 相关参数设置
    cg = Ops(1); 				              % 当前代数
    mg = Ops(3);                              % 最大代数
    bm = Ops(4);                              % 形状参数
    numVar = size(parent, 2) - 1; 	          % 获取变量个数（假设最后一列为适应度）
    mPoint = round(rand * (numVar - 1)) + 1;  % 随机选择一个变量进行突变
    md = round(rand); 			              % 随机选择突变方向，0 表示向下限突变，1 表示向上限突变
    if md 					                  % 向上限突变
      newValue = parent(mPoint) + delta(cg, mg, bounds(mPoint, 2) - parent(mPoint), bm);
    else 					                  % 向下限突变
      newValue = parent(mPoint) - delta(cg, mg, parent(mPoint) - bounds(mPoint, 1), bm);
    end
    parent(mPoint) = newValue; 		          % 更新突变后的变量值
end
