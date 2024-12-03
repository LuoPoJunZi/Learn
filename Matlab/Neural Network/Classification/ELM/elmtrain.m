function [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, N, TF, TYPE)
% ELMTrain - 训练极限学习机模型
%
% 输入参数:
%   p_train - 训练集输入特征矩阵 (R * Q)
%   t_train - 训练集目标输出矩阵 (S * Q)
%   N       - 隐藏层神经元数量
%   TF      - 激活函数类型 ('sig' 或 'hardlim')
%   TYPE    - 回归模式 (0, default) 或 分类模式 (1)
%
% 输出参数:
%   IW  - 输入权重矩阵 (N * R)
%   B   - 偏置向量 (N * 1)
%   LW  - 输出权重矩阵 (N * S)
%   TF  - 激活函数类型
%   TYPE- 模式类型

    % 检查输入数据维度是否匹配
    if size(p_train, 2) ~= size(t_train, 2)
        error('ELM:Arguments', '训练集输入P和输出T的列数必须相同。');
    end

    %% 转入分类模式
    if TYPE == 1
        t_train = ind2vec(t_train);          % 将类别标签转换为独热编码
    end

    %% 初始化权重
    R = size(p_train, 1);                    % 输入特征维度
    Q = size(t_train, 2);                    % 训练样本数量
    IW = rand(N, R) * 2 - 1;                  % 随机初始化输入权重矩阵，范围[-1, 1]
    B  = rand(N, 1);                          % 随机初始化偏置向量
    BiasMatrix = repmat(B, 1, Q);            % 复制偏置向量以匹配样本数量

    %% 计算隐层输出
    tempH = IW * p_train + BiasMatrix;       % 计算隐层输入

    %% 选择激活函数
    switch TF
        case 'sig'
            H = 1 ./ (1 + exp(-tempH));      % Sigmoid激活函数
        case 'hardlim'
            H = hardlim(tempH);              % 硬限制激活函数
        otherwise
            error('Unsupported transfer function.');
    end

    %% 伪逆计算输出权重
    LW = pinv(H') * t_train';                % 计算输出权重矩阵，使用伪逆
end
