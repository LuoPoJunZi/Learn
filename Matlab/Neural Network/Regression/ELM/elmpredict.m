function Y = elmpredict(p_test, IW, B, LW, TF, TYPE)
    % ELM预测函数
    % 输入：
    %   p_test - 测试集输入数据 (R * Q)
    %   IW     - 输入权重矩阵 (N * R)
    %   B      - 偏置向量 (N * 1)
    %   LW     - 输出权重矩阵 (N * S)
    %   TF     - 激活函数类型 ('sig' 或 'hardlim')
    %   TYPE   - 回归或分类类型 (0 为回归, 1 为分类)
    % 输出：
    %   Y      - 预测结果 (1 * Q) 或类别索引
    
    %% 计算隐层输出
    Q = size(p_test, 2);                          % 测试集样本数
    BiasMatrix = repmat(B, 1, Q);                 % 将偏置向量复制Q列，形成偏置矩阵
    tempH = IW * p_test + BiasMatrix;             % 计算隐藏层的线性组合
    
    %% 选择激活函数
    switch TF
        case 'sig'
            H = 1 ./ (1 + exp(-tempH));           % Sigmoid激活函数
        case 'hardlim'
            H = hardlim(tempH);                   % Hardlim激活函数
        otherwise
            error('未知的激活函数类型');
    end
    
    %% 计算输出
    Y = (H' * LW)';                                % 计算输出层的线性组合
    
    %% 转化分类模式
    if TYPE == 1
        temp_Y = zeros(size(Y));                    % 初始化临时矩阵用于存储分类结果
        for i = 1:size(Y, 2)
            [~, index] = max(Y(:, i));              % 找到每列中的最大值索引
            temp_Y(index, i) = 1;                    % 将对应位置设为1
        end
        Y = vec2ind(temp_Y);                         % 将二进制向量转换为类别索引
    end
end
