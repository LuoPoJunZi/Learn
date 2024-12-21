function Y = elmpredict(p_test, IW, B, LW, TF, TYPE)
    % elmpredict: 使用ELM模型进行预测
    % 输入参数：
    %   p_test - 测试集输入特征矩阵 (R × Q)
    %   IW     - 输入权重矩阵 (N × R)
    %   B      - 偏置向量 (N × 1)
    %   LW     - 输出权重矩阵 (S × N)
    %   TF     - 激活函数类型 ('sig' 或 'hardlim')
    %   TYPE   - 预测类型 (0: 回归, 1: 分类)
    % 输出参数：
    %   Y      - 预测结果向量

    %% 计算隐层输出
    Q = size(p_test, 2);                       % 获取测试集样本数量
    BiasMatrix = repmat(B, 1, Q);             % 将偏置向量B重复Q次，形成偏置矩阵
    tempH = IW * p_test + BiasMatrix;          % 计算隐层输入

    %% 选择激活函数
    switch TF
        case 'sig'
            H = 1 ./ (1 + exp(-tempH));        % Sigmoid激活函数
        case 'hardlim'
            H = hardlim(tempH);                % Hardlim激活函数
        otherwise
            error('Unsupported transfer function');
    end

    %% 计算输出
    Y = (H' * LW)';                            % 计算输出层结果

    %% 转化分类模式
    if TYPE == 1
        temp_Y = zeros(size(Y));                % 初始化临时输出矩阵
        for i = 1:size(Y, 2)
            [~, index] = max(Y(:, i));          % 找到每个样本的最大值索引
            temp_Y(index, i) = 1;               % 设置对应位置为1
        end
        Y = vec2ind(temp_Y);                     % 将向量转换为类别索引
    end
end
