function f = evaluate_objective(x)
%% 评估目标函数值
% 输入：
% x - 决策变量向量
% 输出：
% f - 对应目标函数值的向量

% 调用目标函数 ZDT3 来计算目标值
% 如果需要使用不同的目标函数，可以更改为其他函数名称
f = zdt3(x); 

end
