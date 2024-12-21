function [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, N, TF, TYPE)
    % elmtrain: 训练ELM模型
    % 输入参数：
    %   p_train - 训练集输入特征矩阵 (R × Q)
    %   t_train - 训练集目标变量矩阵 (S × Q)
    %   N       - 隐藏层神经元数量
    %   TF      - 激活函数类型 ('sig' 或 'hardlim')
    %   TYPE    - 任务类型 (0: 回归, 1: 分类)
    % 输出参数：
    %   IW  - 输入权重矩阵 (N × R)
    %   B   - 偏置向量 (N × 1)
    %   LW  - 输出权重矩阵 (S × N)
    %   TF  - 激活函数类型
    %   TYPE- 任务类型

    % 检查输入输出样本数量是否一致
    if size(p_train, 2) ~= size(t_train, 2)
        error('ELM:Arguments', '训练集的输入和输出样本数量必须相同。');
    end

    %% 转入分类模式
    if TYPE == 1
        t_train = ind2vec(t_train);             % 将类别标签转换为向量形式
    end

    %% 初始化权重
    R = size(p_train, 1);                       % 输入特征维度
    Q = size(t_train, 2);                       % 样本数量
    IW = rand(N, R) * 2 - 1;                     % 随机初始化输入权重矩阵，范围[-1, 1]
    B  = rand(N, 1);                             % 随机初始化偏置向量，范围[0, 1]
    BiasMatrix = repmat(B, 1, Q);               % 将偏置向量B重复Q次，形成偏置矩阵

    %% 计算隐层输出
    tempH = IW * p_train + BiasMatrix;           % 计算隐层输入

    %% 选择激活函数
    switch TF
        case 'sig'
            H = 1 ./ (1 + exp(-tempH));         % Sigmoid激活函数
        case 'hardlim'
            H = hardlim(tempH);                 % Hardlim激活函数
        otherwise
            error('Unsupported transfer function');
    end

    %% 伪逆计算输出权重
    LW = pinv(H') * t_train';                    % 使用伪逆计算输出权重矩阵，确保最小化预测误差
end
