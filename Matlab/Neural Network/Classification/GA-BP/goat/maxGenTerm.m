function done = maxGenTerm(ops, ~, ~)
% maxGenTerm - 终止函数，当达到最大代数时终止遗传算法
%
% 输入参数:
%   ops    - 选项向量 [当前代数 最大代数]
%   ~      - 占位符参数（未使用）
%   ~      - 占位符参数（未使用）
%
% 输出参数:
%   done - 终止标志，达到最大代数时为 1，否则为 0

    %% 解析参数
    currentGen = ops(1); % 当前代数
    maxGen     = ops(2); % 最大代数
    
    %% 判断是否达到终止条件
    done       = currentGen >= maxGen; % 如果当前代数大于等于最大代数，返回 1，否则返回 0
end
