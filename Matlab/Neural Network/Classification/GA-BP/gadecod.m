function [val, W1, B1, W2, B2] = gadecod(x)
% Gadecod - 解码遗传算法优化后的参数，并训练神经网络
%
% 输入参数:
%   x - 遗传算法优化后的参数向量
%
% 输出参数:
%   val - 适应度值
%   W1  - 输入层到隐藏层的权重矩阵
%   B1  - 隐藏层的偏置向量
%   W2  - 隐藏层到输出层的权重矩阵
%   B2  - 输出层的偏置向量

    %% 读取主空间变量
    S1 = evalin('base', 'S1');             % 读取隐藏层节点个数
    net = evalin('base', 'net');           % 读取网络参数
    p_train = evalin('base', 'p_train');   % 读取训练集输入数据
    t_train = evalin('base', 't_train');   % 读取训练集目标输出数据
    
    %% 参数初始化
    R2 = size(p_train, 1);                 % 输入节点数，即特征数量
    S2 = size(t_train, 1);                 % 输出节点数，即类别数量
    
    %% 输入权重编码
    % 从优化参数向量 x 中提取输入层到隐藏层的权重 W1
    for i = 1 : S1
        for k = 1 : R2
            W1(i, k) = x(R2 * (i - 1) + k);
        end
    end
    
    %% 输出权重编码
    % 从优化参数向量 x 中提取隐藏层到输出层的权重 W2
    for i = 1 : S2
        for k = 1 : S1
            W2(i, k) = x(S1 * (i - 1) + k + R2 * S1);
        end
    end
    
    %% 隐藏层偏置编码
    % 从优化参数向量 x 中提取隐藏层的偏置 B1
    for i = 1 : S1
        B1(i, 1) = x((R2 * S1 + S1 * S2) + i);
    end
    
    %% 输出层偏置编码
    % 从优化参数向量 x 中提取输出层的偏置 B2
    for i = 1 : S2
        B2(i, 1) = x((R2 * S1 + S1 * S2 + S1) + i);
    end
    
    %% 赋值并计算
    net.IW{1, 1} = W1;      % 设置输入层到隐藏层的权重
    net.LW{2, 1} = W2;      % 设置隐藏层到输出层的权重
    net.b{1}     = B1;      % 设置隐藏层的偏置
    net.b{2}     = B2;      % 设置输出层的偏置
    
    %% 模型训练
    net.trainParam.showWindow = 0;      % 关闭训练窗口（避免弹出界面干扰）
    net = train(net, p_train, t_train); % 使用 BP 算法训练神经网络
    
    %% 仿真测试
    t_sim1 = sim(net, p_train);         % 使用训练数据进行仿真测试，得到预测输出
    
    %% 反归一化
    T_train = vec2ind(t_train);         % 将目标输出的独热编码转换为类别索引
    T_sim1  = vec2ind(t_sim1);          % 将预测输出的独热编码转换为类别索引
    
    %% 计算适应度值
    % 适应度值 val 计算公式：1 / (1 - 准确率)
    % 准确率 = 正确预测的样本数 / 总样本数
    % 适应度值越小表示模型性能越好
    val = 1 ./ (1 - sum(T_sim1 == T_train) ./ size(p_train, 2));
end
