%% 主函数：非支配鲸鱼优化算法（NSWOA）
% 清除工作空间和图形窗口
clc
clear
close all

%% 设置问题参数
D = 30; % 决策变量的数量
M = 2; % 目标函数的数量
K = M + D; % 决策变量和目标函数值的总维度
LB = ones(1, D) .* 0; % 决策变量的下界，所有变量的最小值为0
UB = ones(1, D) .* 1; % 决策变量的上界，所有变量的最大值为1
Max_iteration = 100;  % 最大迭代次数
SearchAgents_no = 100; % 种群大小，即搜索代理的数量
ishow = 10; % 每隔多少代显示一次进度

%% 初始化种群
% 调用初始化函数，生成初始种群
chromosome = initialize_variables(SearchAgents_no, M, D, LB, UB);

%% 对初始化的种群进行非支配排序
% 使用非支配排序算法对种群进行排序，划分不同的前沿
intermediate_chromosome = non_domination_sort_mod(chromosome, M, D);

%% 选择操作
% 替换种群，只保留满足要求的个体
Population = replace_chromosome(intermediate_chromosome, M, D, SearchAgents_no);

%% 开始演化过程
% 调用NSWOA主算法进行优化
Pareto = NSWOA(D, M, LB, UB, Population, SearchAgents_no, Max_iteration, ishow);

%% 保存帕累托解集
% 将最终的帕累托前沿保存为文本文件，以供后续使用
save Pareto.txt Pareto -ascii;

%% 绘制结果
% 调用绘图函数，绘制帕累托前沿
plot_data2(M, D, Pareto);
