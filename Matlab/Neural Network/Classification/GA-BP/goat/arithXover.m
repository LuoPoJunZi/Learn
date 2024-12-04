function [C1, C2] = arithXover(P1, P2, ~, ~)
% ArithXover - 算术交叉操作，用于生成两个子代
%
% 输入参数:
%   P1 - 第一个父代个体
%   P2 - 第二个父代个体
%   ~  - 占位符参数（未使用）
%   ~  - 占位符参数（未使用）
%
% 输出参数:
%   C1 - 第一个子代个体
%   C2 - 第二个子代个体

    %% 选择一个随机的混合量
    a = rand; % 生成一个介于 0 和 1 之间的随机数
    
    %% 创建子代
    C1 = P1 * a + P2 * (1 - a); % 使用混合量 a 生成第一个子代
    C2 = P1 * (1 - a) + P2 * a; % 使用混合量 (1 - a) 生成第二个子代
end