%% CostFunction.m 
% J  [OUT] : 目标向量。J 是一个矩阵，行数对应 X 中的试验向量，列数对应目标数量。
% X   [IN] : 决策变量向量。X 是一个矩阵，行数对应试验向量，列数对应决策变量数量。
% Dat [IN] : 在 NNCparam.m 中定义的参数。

%% 作者
% Gilberto Reynoso Meza
% gilreyme@upv.es
% http://cpoh.upv.es/en/gilberto-reynoso-meza.html
% http://www.mathworks.es/matlabcentral/fileexchange/authors/289050

%% 新版本发布及问题修复请访问：
% http://cpoh.upv.es/en/research/software.html
% Matlab Central 文件交换平台

%% 主函数调用
% 该函数根据提供的决策变量 X 和数据 Dat，返回相应的目标向量 J。
function J=CostFunction(X,Dat)

% 判断使用哪个问题的目标函数
if strcmp(Dat.CostProblem,'DTLZ2')
    % 如果问题类型为 DTLZ2，则调用 DTLZ2 函数
    J=DTLZ2(X,Dat);
elseif strcmp(Dat.CostProblem,'YourProblem')
    % 这里可以调用你自己的多目标问题的目标函数
end

%% DTLZ2 基准函数。定义见以下参考文献：
% K. Deb, L. Tiele, M. Laummans 和 E. Zitzler. 
% 进化多目标优化的可扩展测试问题。
% ETH Zurich 通讯网技术研究所，技术报告 No. 112, 2001年2月。
function J=DTLZ2(X,Dat)

% 获取种群规模 Xpop
Xpop=size(X,1);
% 获取变量数量 Nvar 和目标数量 M
Nvar=Dat.NVAR;
M=Dat.NOBJ;
% K 是与变量和目标数量相关的参数
K=Nvar+1-M;
% 初始化目标向量矩阵 J，大小为 Xpop 行 M 列，所有元素初始化为1
J=ones(Xpop,M);

% 对每个种群个体进行计算
for xpop=1:Xpop
    % 计算 Gxm，这是 DTLZ2 的内部辅助函数之一
    Gxm=(X(xpop,M:Nvar)-0.5*ones(1,K))*(X(xpop,M:Nvar)-0.5*ones(1,K))';
    % 计算 Cos，这是余弦函数的乘积，用于目标函数计算
    Cos=cos(X(xpop,1:M-1)*pi/2);

    % 计算第一个目标函数
    J(xpop,1)=prod(Cos)*(1+Gxm);
    
    % 计算其余的目标函数
    for nobj=1:M-1
        J(xpop,nobj+1)=(J(xpop,1)/prod(Cos(1,M-nobj:M-1))) * sin(X(xpop,M-nobj)*pi/2);
    end
end
