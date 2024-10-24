%% MODE
% 基于差分进化 (DE) 的多目标进化算法 (MOEA)。
% 它实现了一种基于纯支配关系的贪心选择。

%% 总体描述
% 此代码实现了基于差分进化算法（DE）的多目标优化算法。
% 当优化一个目标时，标准的 DE 运行；如果优化两个或更多目标，则使用支配关系执行 DE 算法中的贪心选择步骤。

%% 主函数 MODE
function OUT=MODE(MODEDat)

%% 从 MODEDat 中读取参数
Generaciones  = MODEDat.MAXGEN;    % 最大进化代数。
Xpop          = MODEDat.XPOP;      % 种群规模。
Nvar          = MODEDat.NVAR;      % 决策变量的数量。
Nobj          = MODEDat.NOBJ;      % 目标数量。
Bounds        = MODEDat.FieldD;    % 优化边界。
Initial       = MODEDat.Initial;   % 初始化边界。
ScalingFactor = MODEDat.Esc;       % DE 算法中的缩放因子。
CrossOverP    = MODEDat.Pm;        % DE 算法中的交叉概率。
mop           = MODEDat.mop;       % 目标函数（成本函数）。

%% 初始化随机种群
Parent = zeros(Xpop,Nvar);  % 父代种群。
Mutant = zeros(Xpop,Nvar);  % 变异种群。
Child  = zeros(Xpop,Nvar);  % 子代种群。
FES    = 0;                 % 函数评估次数。

% 为每个个体初始化随机决策变量
for xpop=1:Xpop
    for nvar=1:Nvar
        Parent(xpop,nvar) = Initial(nvar,1)+(Initial(nvar,2)... 
                            - Initial(nvar,1))*rand();  % 随机初始化
    end
end

% 如果已存在初始种群，则使用它
if size(MODEDat.InitialPop,1)>=1
    Parent(1:size(MODEDat.InitialPop,1),:)=MODEDat.InitialPop;
end

% 计算父代种群的目标值
JxParent = mop(Parent,MODEDat);
FES = FES + Xpop;  % 更新函数评估次数

%% 进化过程
for n=1:Generaciones 
    for xpop=1:Xpop
        % 随机选择三个不同的个体进行变异
        rev=randperm(Xpop);
        
        %% 变异向量计算
        Mutant(xpop,:)= Parent(rev(1,1),:)+ScalingFactor*... 
                        (Parent(rev(1,2),:)-Parent(rev(1,3),:));
        
        % 确保变异后的个体仍在变量边界内
        for nvar=1:Nvar
            if Mutant(xpop,nvar) < Bounds(nvar,1)
                Mutant(xpop,nvar) = Bounds(nvar,1);
            elseif Mutant(xpop,nvar) > Bounds(nvar,2)
                Mutant(xpop,nvar) = Bounds(nvar,2);
            end
        end
        
        %% 交叉操作
        for nvar=1:Nvar
            if rand() > CrossOverP
                Child(xpop,nvar) = Parent(xpop,nvar);
            else
                Child(xpop,nvar) = Mutant(xpop,nvar);
            end
        end
    end

    % 计算子代种群的目标值
    JxChild = mop(Child,MODEDat);
    FES = FES + Xpop;  % 更新函数评估次数

    %% 选择操作：根据支配关系选择子代或父代个体
    for xpop=1:Xpop
        if JxChild(xpop,:) <= JxParent(xpop,:) 
            Parent(xpop,:) = Child(xpop,:);
            JxParent(xpop,:) = JxChild(xpop,:);
        end
    end
    
    % 保存 Pareto 前沿和 Pareto 集
    PFront = JxParent;
    PSet = Parent;

    % 输出当前种群和目标值
    OUT.Xpop           = Parent;   % 种群
    OUT.Jpop           = JxParent; % 种群的目标值
    OUT.PSet           = PSet;     % Pareto 集
    OUT.PFront         = PFront;   % Pareto 前沿
    OUT.Param          = MODEDat;  % MODE 参数
    MODEDat.CounterGEN = n;        % 记录当前代数
    MODEDat.CounterFES = FES;      % 记录函数评估次数
    
    % 打印和显示当前代数结果
    [OUT MODEDat] = PrinterDisplay(OUT, MODEDat);
    
    % 检查终止条件
    if FES > MODEDat.MAXFUNEVALS || n > MODEDat.MAXGEN
        disp('达到了终止条件。')
        break;
    end
end

% 最终的 Pareto 前沿和 Pareto 集
OUT.Xpop = PSet;
OUT.Jpop = PFront;
[OUT.PFront, OUT.PSet] = DominanceFilter(PFront, PSet);  % 支配过滤

% 如果需要保存结果
if strcmp(MODEDat.SaveResults, 'yes')
    save(['OUT_' datestr(now,30)], 'OUT');  % 保存结果
end

% 显示信息
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
disp('红色星号: 计算的集合。')
disp('黑色菱形: 过滤后的集合。')
if strcmp(MODEDat.SaveResults, 'yes')
    disp(['查看 OUT_' datestr(now,30) ' 文件中的结果。'])
end
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

% 可视化 Pareto 前沿
F = OUT.PFront;
for xpop=1:size(F,1)
    if Nobj == 1
        figure(123); hold on;
        plot(MODEDat.CounterGEN, log(min(F(:,1))), 'dk', ...
            'MarkerFaceColor', 'k'); grid on; hold on;
    elseif Nobj == 2
        figure(123); hold on;
        plot(F(xpop,1), F(xpop,2), 'dk', 'MarkerFaceColor', 'k'); grid on; hold on;
    elseif Nobj == 3
        figure(123); hold on;
        plot3(F(xpop,1), F(xpop,2), F(xpop,3), 'dk', 'MarkerFaceColor', 'k'); grid on; hold on;
    end
end

%% 打印和显示信息
% 可根据需要修改
function [OUT, Dat] = PrinterDisplay(OUT, Dat)
disp('------------------------------------------------')
disp(['代数: ' num2str(Dat.CounterGEN)]);
disp(['评估次数: ' num2str(Dat.CounterFES)]);
disp(['Pareto 前沿大小: ' mat2str(size(OUT.PFront,1))]);
disp('------------------------------------------------')

% 每隔一代可视化 Pareto 前沿
if mod(Dat.CounterGEN,1) == 0
    if Dat.NOBJ == 3
        figure(123);
        plot3(OUT.PFront(:,1), OUT.PFront(:,2), OUT.PFront(:,3), '*r'); 
        grid on;
    elseif Dat.NOBJ == 2
        figure(123);
        plot(OUT.PFront(:,1), OUT.PFront(:,2), '*r'); grid on;
    elseif Dat.NOBJ == 1
        figure(123);
        plot(Dat.CounterGEN, log(min(OUT.PFront(:,1))), '*r'); grid on; hold on;
    end
end

%% 支配过滤
% 基于支配准则的过滤器
function [Frente, Conjunto] = DominanceFilter(F, C)
Xpop = size(F,1);
Nobj = size(F,2);
Nvar = size(C,2);
Frente = zeros(Xpop, Nobj);
Conjunto = zeros(Xpop, Nvar);
k = 0;

for xpop=1:Xpop
    Dominado = 0;  % 假设该个体不被支

配
    
    for compara=1:Xpop
        if F(xpop,:) == F(compara,:)
            if xpop > compara
                Dominado = 1;  % 如果目标值相等，较晚个体被支配
                break;
            end
        else
            if F(xpop,:) >= F(compara,:)  % 如果当前个体被其他个体支配
                Dominado = 1;
                break;
            end
        end
    end
    
    % 如果未被支配，加入前沿集
    if Dominado == 0
        k = k + 1;
        Frente(k,:) = F(xpop,:);
        Conjunto(k,:) = C(xpop,:);
    end
end

% 返回前沿集和决策变量集
Frente = Frente(1:k,:);
Conjunto = Conjunto(1:k,:);
