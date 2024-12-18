function error = fun(pop, hiddennum, net, p_train, t_train)
% FUN 计算粒子对应的误差值，用于PSO优化
% 输入：
%   pop       - 当前粒子的位置（神经网络的权重和阈值）
%   hiddennum - 隐藏层节点数
%   net       - 神经网络对象
%   p_train   - 训练集输入数据
%   t_train   - 训练集目标输出
% 输出：
%   error     - 当前粒子的适应度值（分类错误率）

    %% 节点个数
    inputnum  = size(p_train, 1);  % 输入层节点数
    outputnum = size(t_train, 1);  % 输出层节点数
    
    %% 提取权值和阈值
    % 从粒子位置向量中提取输入层到隐藏层的权重w1
    w1 = pop(1 : inputnum * hiddennum);
    % 提取隐藏层的阈值B1
    B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
    % 提取隐藏层到输出层的权重w2
    w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum);
    % 提取输出层的阈值B2
    B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);
    
    %% 网络赋值
    % 将提取的权重和阈值赋值给神经网络
    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );    % 输入层到隐藏层的权重
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);    % 隐藏层到输出层的权重
    net.b{1}     = reshape(B1, hiddennum, 1);           % 隐藏层的阈值
    net.b{2}     = B2';                                 % 输出层的阈值
    
    %% 网络训练
    net = train(net, p_train, t_train);  % 使用训练集训练网络
    
    %% 仿真测试
    t_sim1 = sim(net, p_train);          % 使用训练集进行仿真预测
    
    %% 反归一化
    T_sim1  = vec2ind(t_sim1 );          % 将网络输出转换为类别索引
    T_train = vec2ind(t_train);          % 将训练集目标输出转换为类别索引
    
    %% 适应度值
    error = 1 - sum(T_sim1 == T_train) / length(T_sim1);  % 计算分类错误率
end
