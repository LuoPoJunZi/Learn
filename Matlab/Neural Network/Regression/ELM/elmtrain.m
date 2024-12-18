function [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, N, TF, TYPE)
    % ELM训练函数
    % 输入：
    %   p_train - 训练集输入数据 (R * Q)
    %   t_train - 训练集输出数据 (S * Q)
    %   N       - 隐藏层神经元数量
    %   TF      - 激活函数类型 ('sig' 或 'hardlim')
    %   TYPE    - 回归或分类类型 (0 为回归, 1 为分类)
    % 输出：
    %   IW  - 输入权重矩阵 (N * R)
    %   B   - 偏置向量 (N * 1)
    %   LW  - 输出权重矩阵 (N * S)
    %   TF  - 激活函数类型
    %   TYPE - 回归或分类类型
    
    % 检查输入输出样本数是否匹配
    if size(p_train, 2) ~= size(t_train, 2)
        error('ELM:Arguments', '训练集的输入和输出样本数必须相同。');
    end
    
    %% 转入分类模式
    if TYPE == 1
        t_train = ind2vec(t_train);                 % 将类别索引转换为二进制向量
    end
    
    %% 初始化权重
    R = size(p_train, 1);                           % 输入特征维度
    Q = size(t_train, 2);                           % 样本数
    IW = rand(N, R) * 2 - 1;                        % 随机初始化输入权重矩阵，范围[-1, 1]
    B  = rand(N, 1);                                % 随机初始化偏置向量，范围[0, 1]
    BiasMatrix = repmat(B, 1, Q);                   % 复制偏置向量，形成偏置矩阵
    
    %% 计算隐藏层输出
    tempH = IW * p_train + BiasMatrix;              % 计算隐藏层的线性组合
    
    %% 选择激活函数
    switch TF
        case 'sig'
            H = 1 ./ (1 + exp(-tempH));            % Sigmoid激活函数
        case 'hardlim'
            H = hardlim(tempH);                    % Hardlim激活函数
        otherwise
            error('未知的激活函数类型');
    end
    
    %% 伪逆计算输出权重
    LW = pinv(H') * t_train';                        % 使用伪逆计算输出权重矩阵
end
