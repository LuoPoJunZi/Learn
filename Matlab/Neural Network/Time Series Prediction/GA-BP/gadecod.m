function [val, W1, B1, W2, B2] = gadecod(x)
    % gadecod: 使用遗传算法解码染色体并训练BP神经网络
    % 输入参数：
    %   x - 染色体向量，编码了BP网络的权重和偏置
    % 输出参数：
    %   val - 适应度值（1/RMSE）
    %   W1  - 输入层到隐藏层的权重矩阵
    %   B1  - 隐藏层的偏置向量
    %   W2  - 隐藏层到输出层的权重矩阵
    %   B2  - 输出层的偏置向量

    %% 读取主空间变量
    S1 = evalin('base', 'S1');             % 读取优化参数个数，即隐藏层节点数
    net = evalin('base', 'net');           % 读取BP神经网络对象
    p_train = evalin('base', 'p_train');   % 读取训练集输入数据
    t_train = evalin('base', 't_train');   % 读取训练集目标数据

    %% 参数初始化
    R2 = size(p_train, 1);                 % 输入节点数，即特征维度
    S2 = size(t_train, 1);                 % 输出节点数，即目标变量维度

    %% 输入权重编码
    W1 = zeros(S1, R2);                     % 初始化输入层到隐藏层的权重矩阵
    for i = 1:S1
        for k = 1:R2
            W1(i, k) = x(R2 * (i - 1) + k); % 从染色体x中提取输入权重
        end
    end

    %% 输出权重编码
    W2 = zeros(S2, S1);                     % 初始化隐藏层到输出层的权重矩阵
    for i = 1:S2
        for k = 1:S1
            W2(i, k) = x(S1 * (i - 1) + k + R2 * S1); % 从染色体x中提取输出权重
        end
    end

    %% 隐藏层偏置编码
    B1 = zeros(S1, 1);                      % 初始化隐藏层的偏置向量
    for i = 1:S1
        B1(i, 1) = x((R2 * S1 + S1 * S2) + i); % 从染色体x中提取隐藏层偏置
    end

    %% 输出层偏置编码
    B2 = zeros(S2, 1);                      % 初始化输出层的偏置向量
    for i = 1:S2
        B2(i, 1) = x((R2 * S1 + S1 * S2 + S1) + i); % 从染色体x中提取输出层偏置
    end

    %% 赋值并计算
    net.IW{1, 1} = W1;                       % 设置BP网络的输入权重矩阵
    net.LW{2, 1} = W2;                       % 设置BP网络的输出权重矩阵
    net.b{1}     = B1;                       % 设置BP网络的隐藏层偏置
    net.b{2}     = B2;                       % 设置BP网络的输出层偏置

    %% 模型训练
    net.trainParam.showWindow = 0;           % 关闭训练窗口，避免在GA优化过程中弹出训练界面
    net = train(net, p_train, t_train);      % 使用训练集数据训练BP神经网络

    %% 仿真测试
    t_sim1 = sim(net, p_train);              % 使用训练集数据进行仿真预测，得到训练集预测结果

    %% 计算适应度值
    val = 1 ./ (sqrt(sum((t_sim1 - t_train).^2) ./ length(t_sim1))); % 适应度值为1/RMSE，R2高时val高
end
