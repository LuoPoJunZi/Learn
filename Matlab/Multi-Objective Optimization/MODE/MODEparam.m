%% MODEparam
% 生成运行多目标差分进化 (MODE) 优化算法所需的参数。
% 差分进化算法 (DE) 是一种用于全局优化的简单而有效的启发式算法：

%% 总体描述
% 此代码实现了基于差分进化算法 (DE) 的多目标优化算法：
% 当只优化一个目标时，运行标准的 DE 算法；如果优化两个或多个目标，DE 算法中的贪心选择步骤将使用支配关系进行。

%% 清理工作区环境
clear all;
close all;
clc;

%% 优化问题相关变量设置

MODEDat.NOBJ = 2;                          % 目标数量
MODEDat.NRES = 0;                          % 约束数量
MODEDat.NVAR   = 10;                       % 决策变量的数量
MODEDat.mop = str2func('CostFunction');    % 成本函数，引用外部的成本函数文件
MODEDat.CostProblem = 'DTLZ2';             % 成本函数实例，使用 DTLZ2 问题作为测试案例

% 优化边界设置
MODEDat.FieldD = [zeros(MODEDat.NVAR,1)...  % 决策变量的下边界
                  ones(MODEDat.NVAR,1)];    % 决策变量的上边界
MODEDat.Initial = [zeros(MODEDat.NVAR,1)... % 优化初始下边界
                   ones(MODEDat.NVAR,1)];   % 优化初始上边界

%% 优化算法相关变量设置
% 参数调优指导文献：
%
% Storn, R., Price, K., 1997. 差分进化：一种用于连续空间全局优化的简单且有效的启发式算法。
% 全球优化杂志 11, 341-359。
%
% Das, S., Suganthan, P. N., 2010. 差分进化：现状综述。IEEE 进化计算事务，第 15 卷，4-31。

MODEDat.XPOP = 5 * MODEDat.NOBJ;            % 种群规模，设置为目标数量的五倍
MODEDat.Esc = 0.5;                          % 差分进化算法的缩放因子
MODEDat.Pm = 0.2;                           % 交叉概率

%% 其他变量
%
MODEDat.InitialPop = [];                    % 初始种群（如果有的话）
MODEDat.MAXGEN = 10000;                     % 最大代数（进化的最大次数）
MODEDat.MAXFUNEVALS = 150 * MODEDat.NVAR * MODEDat.NOBJ;  % 最大函数评估次数
MODEDat.SaveResults = 'yes';                % 若希望在优化过程结束后保存结果，设置为 'yes'，否则设置为 'no'

%% 初始化（请勿修改）
MODEDat.CounterGEN = 0;  % 当前代数计数器
MODEDat.CounterFES = 0;  % 函数评估次数计数器

%% 如果需要，可以在此处放置额外的变量
%
%

%% 运行算法
OUT = MODE(MODEDat);  % 调用 MODE 算法，并将参数传递给它
