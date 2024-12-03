% Main.m
% 非支配排序的蜣螂算法（NSDBO）的主程序
% 该程序初始化种群，运行NSDBO算法，并计算相关评价指标
% 输出:
%   Obtained_Pareto - 通过NSDBO算法获得的Pareto前沿解集
%   X               - Pareto前沿解集对应的决策变量位置

close all;    % 关闭所有打开的图形窗口
clear;        % 清除工作区中的所有变量
clc;          % 清空命令窗口

%% 设置测试问题
TestProblem = 31;                % 测试问题编号（范围1-47）
MultiObj = GetFunInfo(TestProblem); % 获取测试问题的详细信息
MultiObjFnc = MultiObj.name;     % 获取测试问题的名称

%% 设置算法参数
params.Np = 300;        % 种群规模（Population size）
params.Nr = 300;        % 仓库规模（Repository size）
params.maxgen = 300;    % 最大代数（Maximum number of generations）

numOfObj = MultiObj.numOfObj; % 目标函数个数
D = MultiObj.nVar;            % 决策变量维数

%% 运行NSDBO算法
f = NSDBO(params, MultiObj);     % 执行NSDBO算法，返回最终仓库中的个体

%% 提取结果
X = f(:, 1:D);                        % 提取仓库中个体的决策变量位置
Obtained_Pareto = f(:, D+1 : D+numOfObj); % 提取仓库中个体的目标函数值

%% 计算评价指标
if isfield(MultiObj, 'truePF')       % 判断是否存在真实Pareto前沿
    True_Pareto = MultiObj.truePF;    % 获取真实Pareto前沿
    
    %% 计算评价指标
    % ResultData的值分别是IGD、GD、HV、Spacing
    % 其中HV（超体积）越大越好，其他指标（IGD、GD、Spacing）越小越好
    ResultData = [
        IGD(Obtained_Pareto, True_Pareto),     % 反向广义差距
        GD(Obtained_Pareto, True_Pareto),      % 广义差距
        HV(Obtained_Pareto, True_Pareto),      % 超体积
        Spacing(Obtained_Pareto)               % 解集分布间距
    ];
else
    % 如果没有真实Pareto前沿，只计算Spacing
    % Spacing越小说明解集分布越均匀
    ResultData = Spacing(Obtained_Pareto);    % 计算Spacing
end

%% 显示结果
disp('Repository fitness values are stored in Obtained_Pareto');
disp('Repository particles positions are stored in X');
