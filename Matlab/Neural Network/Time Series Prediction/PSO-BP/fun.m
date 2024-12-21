function error = fun(pop, hiddennum, net, p_train, t_train)
% FUN 适应度函数，用于评估粒子群优化算法中每个粒子的适应度
% 输入参数：
%   pop       - 当前粒子的编码（权重和偏置的向量）
%   hiddennum - 隐藏层节点数
%   net       - BP神经网络模型
%   p_train   - 训练集输入特征
%   t_train   - 训练集目标变量
% 输出参数：
%   error     - 当前粒子的适应度值（RMSE）

    %% 节点个数
    inputnum  = size(p_train, 1);   % 输入层节点数
    outputnum = size(t_train, 1);   % 输出层节点数
    
    %% 提取权值和阈值
    w1 = pop(1 : inputnum * hiddennum);  % 提取输入层到隐藏层的权重向量
    B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum); % 提取隐藏层的偏置向量
    w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum); % 提取隐藏层到输出层的权重向量
    B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum); % 提取输出层的偏置向量
     
    %% 网络赋值
    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );  % 将权重向量w1重塑为隐藏层权重矩阵
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum); % 将权重向量w2重塑为输出层权重矩阵
    net.b{1}     = reshape(B1, hiddennum, 1);        % 将偏置向量B1重塑为隐藏层偏置
    net.b{2}     = B2';                               % 将偏置向量B2转置为输出层偏置
    
    %% 网络训练
    net = train(net, p_train, t_train); % 使用训练集数据训练BP神经网络，更新网络权重和偏置
    
    %% 仿真测试
    t_sim1 = sim(net, p_train);  % 使用训练集数据进行仿真预测，得到训练集预测结果
    
    %% 适应度值
    error = sqrt(sum((t_sim1 - t_train) .^ 2) ./ length(t_sim1)); % 计算均方根误差（RMSE）作为适应度值
end
