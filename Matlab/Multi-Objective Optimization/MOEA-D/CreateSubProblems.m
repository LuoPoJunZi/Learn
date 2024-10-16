% 创建子问题的函数
% 输入参数:
% nObj - 目标函数的数量
% nPop - 种群数量
% T - 每个子问题的邻居数量

function sp = CreateSubProblems(nObj, nPop, T)

    % 初始化空的子问题结构体
    empty_sp.lambda = [];       % 存储子问题的权重向量
    empty_sp.Neighbors = [];    % 存储邻居子问题的索引

    % 使用空结构体复制生成种群数量的子问题
    sp = repmat(empty_sp, nPop, 1);
    
    % theta = linspace(0, pi/2, nPop); % 可选：均匀分布的角度，未使用

    for i = 1:nPop
        % 随机生成权重向量并进行归一化
        lambda = rand(nObj, 1);   
        lambda = lambda / norm(lambda);  % 归一化处理
        sp(i).lambda = lambda;  % 将权重向量存储到子问题中
        
        % 可选：使用角度生成权重向量，未使用
        % sp(i).lambda = [cos(theta(i))
        %                  sin(theta(i))];
    end

    % 从子问题中提取权重向量并计算距离矩阵
    LAMBDA = [sp.lambda]';  
    D = pdist2(LAMBDA, LAMBDA);  % 计算权重向量之间的欧氏距离
    
    for i = 1:nPop
        % 对距离进行排序，获取最近的T个邻居
        [~, SO] = sort(D(i, :));
        sp(i).Neighbors = SO(1:T);  % 存储邻居索引
    end

end
