% 函数 GetGridIndex
% 该函数用于获取粒子在超立方体网格中的索引。网格索引用于确定粒子属于哪个网格单元，
% 以便在多目标优化算法中进行存储和选择操作。
% 参数：
%   - particle: 包含目标值 (Cost) 的粒子
%   - G: 网格结构体数组，其中每个元素包含当前维度的网格边界
% 返回：
%   - Index: 粒子的全局网格索引，用于标识粒子在哪个网格单元中
%   - SubIndex: 每个目标的子网格索引，用于标识粒子在每个目标维度的网格位置

function [Index, SubIndex] = GetGridIndex(particle, G)

    % 获取粒子的目标值向量 (Cost)
    c = particle.Cost;
    
    % 获取目标的数量 (nobj) 和网格单元的数量 (ngrid)
    nobj = numel(c);
    ngrid = numel(G(1).Upper);
    
    % 初始化生成网格索引的表达式，使用 MATLAB 函数 sub2ind
    str = ['sub2ind(' mat2str(ones(1,nobj) * ngrid)];
    
    % 初始化子索引数组 SubIndex，用于存储每个目标的网格索引
    SubIndex = zeros(1, nobj);
    
    % 对每个目标维度进行处理
    for j = 1:nobj
        
        % 获取第 j 个目标的网格边界 (Upper)
        U = G(j).Upper;
        
        % 找到目标值 c(j) 所对应的网格单元的索引
        i = find(c(j) < U, 1, 'first');
        
        % 将该维度的网格索引存储到 SubIndex 数组中
        SubIndex(j) = i;
        
        % 将该维度的索引 i 添加到 str 表达式中，用于计算全局索引
        str = [str ',' num2str(i)];
    end
    
    % 完成用于计算全局索引的表达式
    str = [str ');'];
    
    % 通过 eval 函数执行生成的字符串，计算粒子的全局网格索引
    Index = eval(str);
    
end
