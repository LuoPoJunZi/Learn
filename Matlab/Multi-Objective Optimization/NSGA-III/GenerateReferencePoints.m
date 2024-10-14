function Zr = GenerateReferencePoints(M, p)
    % GenerateReferencePoints 函数生成用于多目标优化中的参考点
    % 输入：
    %   M - 目标函数的数量
    %   p - 参考点划分数，控制参考点的密度
    % 输出：
    %   Zr - 生成的参考点矩阵，每列代表一个参考点

    % 调用 GetFixedRowSumIntegerMatrix 函数生成一个元素和固定的整数矩阵
    % 并对结果进行转置和归一化，得到最终的参考点矩阵 Zr
    Zr = GetFixedRowSumIntegerMatrix(M, p)' / p;
end

function A = GetFixedRowSumIntegerMatrix(M, RowSum)
    % GetFixedRowSumIntegerMatrix 函数递归生成一个矩阵，
    % 其中每一行的元素之和为指定的 RowSum
    % 输入：
    %   M - 行的维数（通常是目标函数的数量）
    %   RowSum - 行元素之和
    % 输出：
    %   A - 生成的整数矩阵，每行元素之和固定为 RowSum

    % 检查 M 是否有效，M 必须为正整数
    if M < 1
        error('M cannot be less than 1.');  % 抛出错误：M 不能小于 1
    end
    
    % 检查 M 是否为整数
    if floor(M) ~= M
        error('M must be an integer.');  % 抛出错误：M 必须是整数
    end
    
    % 基本情况：当 M 为 1 时，返回 RowSum
    if M == 1
        A = RowSum;
        return;
    end

    % 初始化 A 为一个空矩阵
    A = [];
    
    % 递归调用，生成所有可能的行组合，使得每行元素之和为 RowSum
    for i = 0:RowSum
        % 对 M-1 维进行递归，RowSum 减去当前的 i 值
        B = GetFixedRowSumIntegerMatrix(M - 1, RowSum - i);
        
        % 将当前的 i 值扩展到对应矩阵的列前，构成新的一行
        A = [A; i * ones(size(B, 1), 1), B];
    end
end
