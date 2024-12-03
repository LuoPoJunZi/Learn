function change = delta(ct, mt, y, b)
% delta - 计算非均匀突变的变化量
%
% 输入参数:
%   ct - 当前代数
%   mt - 最大代数
%   y  - 最大变化量，即参数值到边界的距离
%   b  - 形状参数
%
% 输出参数:
%   change - 突变的变化量

    %% 计算当前代与最大代的比率
    r = ct / mt;
    if(r > 1)
      r = 0.99; % 防止比率超过 1
    end
    % 使用非均匀分布计算变化量
    change = y * (rand * (1 - r)) ^ b;
end
