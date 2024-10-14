function [pop, d, rho] = AssociateToReferencePoint(pop, params)
    % AssociateToReferencePoint 函数的作用是将种群中的个体与参考点关联起来，
    % 并计算每个个体到关联参考点的距离，以及每个参考点关联的个体数量。

    % 输入：
    %   pop - 种群（包含多个个体，每个个体有其对应的目标函数值）
    %   params - 参数结构体，包含参考点 Zr 和参考点的数量 nZr
    
    % 输出：
    %   pop - 更新后的种群，包含每个个体的关联参考点和到参考点的距离
    %   d - 每个个体到所有参考点的距离矩阵
    %   rho - 每个参考点关联的个体数量

    Zr = params.Zr;   % 参考点矩阵，每一列是一个参考点
    nZr = params.nZr; % 参考点的数量
    
    % 初始化 rho，记录每个参考点关联的个体数量
    rho = zeros(1,nZr);
    
    % 初始化距离矩阵 d，记录每个个体到各个参考点的距离
    d = zeros(numel(pop), nZr);
    
    % 遍历种群中的每个个体
    for i = 1:numel(pop)
        % 遍历每个参考点
        for j = 1:nZr
            % 归一化参考点方向向量 w
            w = Zr(:,j)/norm(Zr(:,j));
            % 获取当前个体的归一化目标值 z
            z = pop(i).NormalizedCost;
            % 计算个体到当前参考点的距离（即垂直距离）
            d(i,j) = norm(z - w'*z*w);
        end
        
        % 找到距离最小的参考点，即个体关联的参考点
        [dmin, jmin] = min(d(i,:));
        
        % 记录个体关联的参考点索引
        pop(i).AssociatedRef = jmin;
        % 记录个体到关联参考点的最小距离
        pop(i).DistanceToAssociatedRef = dmin;
        % 增加该参考点的关联个体数量
        rho(jmin) = rho(jmin) + 1;
    end

end
