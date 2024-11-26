% -------------------------------------------------------------
% 文件名: RouletteWheelSelection.m
% 功能: 实现轮盘赌选择算法，根据给定的权重选择一个个体的索引
%       选择概率与权重成正比，常用于基于适应度的选择
% 输入:
%       weights - 个体的权重数组（正数）
% 输出:
%       choice  - 被选中的个体的索引（整数）
% 作者: [您的名字]
% 日期: [日期]
% -------------------------------------------------------------

function choice = RouletteWheelSelection(weights)
    % 计算权重的累积和，用于构建轮盘赌
    accumulation = cumsum(weights);
    
    % 生成一个介于0和累积和之间的随机数
    p = rand() * accumulation(end);
    
    % 初始化选择结果为-1，表示未选择任何个体
    chosen_index = -1;
    
    % 遍历累积和，找到第一个累积和大于随机数p的位置
    for index = 1:length(accumulation)
        if (accumulation(index) > p)
            chosen_index = index;
            break;
        end
    end
    
    % 如果没有找到合适的索引，默认选择最后一个个体
    if chosen_index == -1
        chosen_index = length(accumulation);
    end
    
    % 将选择结果赋值给输出变量
    choice = chosen_index;
end
