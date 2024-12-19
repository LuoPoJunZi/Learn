function error = fun(pop, hiddennum, net, p_train, t_train)
    % PSO-BP回归的适应度评估函数
    % 输入：
    %   pop        - 粒子位置向量，包含BP网络的所有权重和偏置
    %   hiddennum  - 隐藏层神经元个数
    %   net        - BP神经网络对象
    %   p_train    - 训练集输入数据
    %   t_train    - 训练集输出数据
    % 输出：
    %   error      - 适应度值，基于训练集的预测误差

    %% 节点个数
    inputnum  = size(p_train, 1);  % 输入层节点数（特征维度）
    outputnum = size(t_train, 1);  % 输出层节点数（目标维度）

    %% 提取权值和偏置
    % 从粒子位置向量中提取输入权重w1、隐层偏置B1、输出权重w2和输出层偏置B2
    w1 = pop(1 : inputnum * hiddennum);  % 输入权重向量
    B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);  % 隐层偏置向量
    w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum);  % 输出权重向量
    B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);  % 输出层偏置向量

    %% 网络赋值
    % 将提取的权重和偏置赋值给BP神经网络对象
    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);  % 输入权重矩阵，尺寸为(hiddennum × inputnum)
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum); % 输出权重矩阵，尺寸为(outputnum × hiddennum)
    net.b{1}     = reshape(B1, hiddennum, 1);        % 隐层偏置向量，尺寸为(hiddennum × 1)
    net.b{2}     = B2';                               % 输出层偏置向量，尺寸为(1 × outputnum)

    %% 网络训练
    net = train(net, p_train, t_train);  % 使用训练集数据训练BP神经网络

    %% 仿真测试
    t_sim1 = sim(net, p_train);          % 使用训练集数据进行仿真预测，得到训练集预测结果

    %% 适应度值
    % 计算训练集预测误差的总和，作为适应度值
    error = sum(sqrt(sum((t_sim1 - t_train) .^ 2) ./ length(t_sim1)));
end
