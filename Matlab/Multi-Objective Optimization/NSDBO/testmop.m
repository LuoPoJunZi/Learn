% testmop.m
% 测试多目标优化问题生成器
% 该脚本定义了多个常用的多目标优化测试问题，包括ZDT、DTLZ、WFG系列以及CEC2009的UF和CF函数。
% 通过输入测试问题的名称和维度，可以生成相应的测试问题结构体，包含目标函数、决策变量范围等信息。
%
% 输入:
%   testname    : (字符数组) 测试问题的名称，如 'zdt1', 'dtlz2', 'wfg1', 'uf1', 'cf1' 等。
%   dimension   : (整数) 决策变量的维数。
%                - 对于DTLZ问题，维数 = M - 1 + k；
%                - 对于WFG问题，维数 = l + k；
%
% 全局输入: 注意在选择对应的测试问题时，必须赋值以下全局变量，这些变量在函数中被标记为关键字 'global'。
%   动态问题:
%     itrCounter - 当前迭代次数
%     step       - 动态步长
%     window     - 动态窗口数量
%   WFG问题:
%     k - 与位置相关的参数数量
%     M - 目标函数数量
%     l - 与距离相关的参数数量
%   DTLZ问题:
%     M - 目标函数数量
%     k - 控制维数的参数
%
% 输出:
%   mop         : (结构体) 包含测试问题的详细信息
%                 - name   : 测试问题名称
%                 - od     : 目标维数（Objective Dimension）
%                 - pd     : 决策变量维数（Decision Dimension）
%                 - domain : 决策变量的边界约束（决策边界）
%                 - func   : 目标函数的句柄

function mop = testmop(testname, dimension)
    % 初始化测试问题结构体
    mop = struct('name',[],'od',[],'pd',[],'domain',[],'func',[]);
    
    % 动态调用对应的测试问题生成器函数
    eval(['mop=',testname,'(mop,',num2str(dimension),');']);
end

%% ----------Stationary Multi-Objective Benchmark----------
% ----------ZDT系列。参考文献：[2]----------- 

%% ZDT1 函数生成器
function p = zdt1(p, dim)
    p.name = 'ZDT1';
    p.pd = dim;        % 决策变量维数
    p.od = 2;          % 目标函数维数
    p.domain = [zeros(dim,1) ones(dim,1)]; % 决策变量范围 [0,1]^dim
    p.func = @evaluate; % 目标函数句柄
    
    % ZDT1 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        y(1) = x(1);
        su = sum(x) - x(1);
        g = 1 + 9 * su / (dim - 1);
        y(2) = g * (1 - sqrt(x(1) / g));
    end
end

%% ZDT2 函数生成器
function p = zdt2(p, dim)
    p.name = 'ZDT2';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % ZDT2 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 9 * sum(x(2:dim)) / (dim - 1);
        y(1) = x(1);
        y(2) = g * (1 - (x(1)/g)^2);
    end
end

%% ZDT3 函数生成器
function p = zdt3(p, dim)
    p.name = 'ZDT3';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % ZDT3 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 9 * sum(x(2:dim)) / (dim - 1);
        y(1) = x(1);
        y(2) = g * (1 - sqrt(x(1)/g) - (x(1)/g) * sin(10 * pi * x(1)));
    end
end

%% ZDT4 函数生成器
function p = zdt4(p, dim)
    p.name = 'ZDT4';
    p.pd = dim;
    p.od = 2;
    % 决策变量范围：
    % x1 ∈ [0,1]
    % x2, ..., xdim ∈ [-5,5]
    p.domain = [0 * ones(dim,1) 1 * ones(dim,1)];
    p.domain(1,1) = 0; % x1的下界
    p.domain(1,2) = 1; % x1的上界
    p.func = @evaluate;
    
    % ZDT4 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 10 * (10 - 1);
        for i = 2:10
            g = g + x(i)^2 - 10 * cos(4 * pi * x(i));
        end
        y(1) = x(1);
        y(2) = g * (1 - sqrt(x(1)/g));
    end
end

%% ZDT6 函数生成器
function p = zdt6(p, dim)
    p.name = 'ZDT6';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % ZDT6 评价函数
    function y = evaluate(x)
        y = zeros(2,1);
        g = 1 + 9 * (sum(x(2:dim)) / (dim - 1))^0.25;
        y(1) = 1 - exp(-4 * x(1)) * sin(6 * pi * x(1))^6;
        y(2) = g * (1 - (y(1)/g)^2);
    end
end

%% --------------DTLZ 基准测试-------参考文献：[3]----
% DTLZ系列问题定义

%% DTLZ1 函数生成器
% 建议参数：k = 5 
function p = DTLZ1(p, dim)
    global M k;
    p.name = 'DTLZ1';
    p.pd = dim;      % 决策变量维数
    p.od = M;        % 目标函数维数
    p.domain = [zeros(dim,1) ones(dim,1)]; % 决策变量范围 [0,1]^dim
    p.func = @evaluate;
    
    % DTLZ1 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 5 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = 100 * (k + sum((xm - 0.5).^2 - cos(20 * pi * (xm - 0.5)), 1));

        % 计算目标函数
        f(1,:) = 0.5 * prod(x(1:M-1,:), 1) .* (1 + g);
        for ii = 2:M-1
            f(ii,:) = 0.5 * prod(x(1:M-ii,:), 1) .* (1 - x(M-ii+1,:)) .* (1 + g);
        end
        f(M,:) = 0.5 * (1 - x(1,:)) .* (1 + g);
        y = f;
    end
end

%% DTLZ2 函数生成器
% 建议参数：k = 10
function p = DTLZ2(p, dim)
    global M k;
    p.name = 'DTLZ2';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ2 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum((xm - 0.5).^2, 1);

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-ii,:)), 1) .* sin(pi/2 * x(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(pi/2 * x(1,:));
        y = f;
    end
end

%% DTLZ3 函数生成器
% 建议参数：k = 10
function p = DTLZ3(p, dim)
    global M k;
    p.name = 'DTLZ3';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ3 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = 100 * (k + sum((xm - 0.5).^2 - cos(20 * pi * (xm - 0.5)), 1));

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-ii,:)), 1) .* sin(pi/2 * x(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(pi/2 * x(1,:));
        y = f;
    end
end

%% DTLZ4 函数生成器
% 建议参数：k = 10
function p = DTLZ4(p, dim)
    global M k;
    p.name = 'DTLZ4';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ4 评价函数
    function y = evaluate(x)
        x = x';
        alpha = 100; % 形状参数
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum((xm - 0.5).^2, 1);

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-1,:).^alpha), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(pi/2 * x(1:M-ii,:).^alpha), 1) .* sin(pi/2 * x(M-ii+1,:).^alpha);
        end
        f(M,:) = (1 + g) .* sin(pi/2 * x(1,:).^alpha);
        y = f;
    end
end

%% DTLZ5 函数生成器
% 建议参数：k = 10
function p = DTLZ5(p, dim)
    global M k;
    p.name = 'DTLZ5';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ5 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum((xm - 0.5).^2, 1); 

        % 计算 theta
        theta = pi/2 * x(1,:);
        gr = g(ones(M-2,1), :); % 复制 g 以进行后续运算
        theta(2:M-1,:) = pi ./ (4 * (1 + gr)) .* (1 + 2 * gr .* x(2:M-1,:));

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(theta(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(theta(1:M-ii,:)), 1) .* sin(theta(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(theta(1,:));
        y = f;
    end
end

%% DTLZ6 函数生成器
% 建议参数：k = 10
function p = DTLZ6(p, dim)
    global M k;
    p.name = 'DTLZ6';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ6 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 10 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = sum(xm.^0.1, 1); 

        % 计算 theta
        theta = pi/2 * x(1,:);
        gr = g(ones(M-2,1), :); % 复制 g 以进行后续运算
        theta(2:M-1,:) = pi ./ (4 * (1 + gr)) .* (1 + 2 * gr .* x(2:M-1,:));

        % 计算目标函数
        f(1,:) = (1 + g) .* prod(cos(theta(1:M-1,:)), 1);
        for ii = 2:M-1
            f(ii,:) = (1 + g) .* prod(cos(theta(1:M-ii,:)), 1) .* sin(theta(M-ii+1,:));
        end
        f(M,:) = (1 + g) .* sin(theta(1,:));
        y = f;
    end
end

%% DTLZ7 函数生成器
% 建议参数：k = 20
function p = DTLZ7(p, dim)
    global M k;
    p.name = 'DTLZ7';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % DTLZ7 评价函数
    function y = evaluate(x)
        x = x';
        n = (M - 1) + k; % 默认维数
        if size(x,1) ~= n
            error(['使用 k = 20 时，维数必须为 n = (M - 1) + k = %d。'], n);
        end

        % 计算 g 函数
        xm = x(n - k + 1:end, :); % 最后 k 个变量
        g = 1 + 9/k * sum(xm, 1);

        % 计算前 M-1 个目标函数
        f(1:M-1,:) = x(1:M-1,:);
        
        % 计算最后一个目标函数
        gaux = g(ones(M-1,1), :); % 复制 g 以进行后续运算
        h = M - sum(f ./ (1 + gaux) .* (1 + sin(3 * pi * f)), 1);
        f(M,:) = (1 + g) .* h;
        y = f;
    end
end

%% --------------WFG 基准测试--------参考文献：[1]----
% WFG系列问题定义

%% WFG1 函数生成器
% 决策变量维数：dim = k + l;
function p = wfg1(p, dim)
    global k l M;
    p.name = 'WFG1';
    p.pd = dim;        % 决策变量维数
    p.od = M;          % 目标函数维数
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG1 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 1);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：b_flat
        Ybarbar = Ybar;
        biasA = 0.8;
        biasB = 0.75;
        biasC = 0.85;
        Ybarbar(:, lLoop:n) = b_flat(Ybarbar(:, lLoop:n), biasA, biasB, biasC);
        
        % 应用第三变换：b_poly
        Ybarbarbar = Ybarbar;
        biasA = 0.02;
        Ybarbarbar = b_poly(Ybarbarbar, biasA);
        
        % 应用第四变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            weights = 2 * (lBnd:uBnd);
            T(:,i) = r_sum(Ybarbarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbarbar(:, lLoop:n), 2*(lLoop:n));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(1, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        fM = h_mixed(X(:,1), 1, 5);
        F = h_convex(X(:,1:uLoop));
        F(:,M) = fM;
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* F;
        y = F';
    end
end

%% WFG2 函数生成器
function p = wfg2(p, dim)
    global k l M;
    p.name = 'WFG2';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG2 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 2);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：r_nonsep
        Ybarbar = Ybar;
        uLoop = k + l/2;
        for i = lLoop:uLoop
            lBnd = k + 2*(i - k) - 1;
            uBnd = k + 2*(i - k);
            Ybarbar(:,i) = r_nonsep(Ybar(:, lBnd:uBnd), 2);
        end
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:k + l/2), ones(1, (k + l/2) - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(2, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        fM = h_disc(X(:,1), 1, 1, 5);
        F = h_convex(X(:,1:uLoop));
        F(:,M) = fM;
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* F;
        y = F';
    end
end

%% WFG3 函数生成器
function p = wfg3(p, dim)
    global k l M;
    p.name = 'WFG3';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG3 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 3);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：r_nonsep
        Ybarbar = Ybar;
        uLoop = k + l/2;
        for i = lLoop:uLoop
            lBnd = k + 2*(i - k) - 1;
            uBnd = k + 2*(i - k);
            Ybarbar(:,i) = r_nonsep(Ybar(:, lBnd:uBnd), 2);
        end
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:k + l/2), ones(1, (k + l/2) - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(3, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_linear(X(:,1:uLoop));
        y = F';
    end
end

%% WFG4 函数生成器
function p = wfg4(p, dim)
    global k l M;    
    p.name = 'WFG4';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG4 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 4);
        testNo = 5;

        % 应用第一变换：根据 testNo 选择不同的变换
        if testNo == 4
            shiftA = 30;
            shiftB = 10;
            shiftC = 0.35;
            Ybar = s_multi(Y, shiftA, shiftB, shiftC);
        else
            shiftA = 0.35;
            shiftB = 0.001;
            shiftC = 0.05;
            Ybar = s_decep(Y, shiftA, shiftB, shiftC);
        end
        
        % 应用第二变换：r_sum
        T = NaN * ones(noSols, M);
        lLoop = k + 1;
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(4, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG5 函数生成器
function p = wfg5(p, dim)
    global k l M;
    p.name = 'WFG5';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG5 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 5);

        % 应用第一变换：s_decep
        shiftA = 0.35;
        shiftB = 0.001;
        shiftC = 0.05;
        Ybar = s_decep(Y, shiftA, shiftB, shiftC);
        
        % 应用第二变换：r_sum
        T = NaN * ones(noSols, M);
        lLoop = k + 1;
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(5, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG6 函数生成器
function p = wfg6(p, dim)
    global k l M;
    p.name = 'WFG6';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG6 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 6);

        % 应用第一变换：s_linear
        Ybar = Y;
        lLoop = k + 1;
        shiftA = 0.35;
        Ybar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第二变换：r_nonsep
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_nonsep(Ybar(:, lBnd:uBnd), k/(M-1));
        end
        T(:,M) = r_nonsep(Ybar(:, k+1:k+l), l);
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(6, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG7 函数生成器
function p = wfg7(p, dim)
    global k l M;
    p.name = 'WFG7';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG7 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 7);

        % 应用第一变换：b_param
        Ybar = Y;
        biasA = 0.98 / 49.98;
        biasB = 0.02;
        biasC = 50;
        for i = 1:k
            Ybar(:,i) = b_param(Y(:,i), r_sum(Y(:,i+1:n), ones(1, n-i)), biasA, biasB, biasC);
        end
        
        % 应用第二变换：s_decep 和 s_multi
        Ybarbar = Ybar;
        lLoop = k + 1;
        shiftA = 0.35;
        shiftB = 0.001;
        shiftC = 0.05;
        Ybarbar(:, lLoop:n) = s_decep(Ybar(:, lLoop:n), shiftA, shiftB, shiftC);
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(7, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG8 函数生成器
function p = wfg8(p, dim)
    global k l M;
    p.name = 'WFG8';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG8 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 8);

        % 应用第一变换：b_param
        Ybar = Y;
        lLoop = k + 1;
        biasA = 0.98 / 49.98;
        biasB = 0.02;
        biasC = 50;
        for i = lLoop:n
            Ybar(:,i) = b_param(Y(:,i), r_sum(Y(:,1:i-1), ones(1, i-1)), biasA, biasB, biasC);
        end
        
        % 应用第二变换：s_linear
        Ybarbar = Ybar;
        shiftA = 0.35;
        Ybarbar(:, lLoop:n) = s_linear(Ybar(:, lLoop:n), shiftA);
        
        % 应用第三变换：r_sum
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybarbar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybarbar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(8, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG9 函数生成器
function p = wfg9(p, dim)
    global k l M;
    p.name = 'WFG';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG9 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 9);

        % 应用第一变换：b_param
        Ybar = Y;
        uLoop = n - 1;
        biasA = 0.98 / 49.98;
        biasB = 0.02;
        biasC = 50;
        for i = 1:uLoop
            Ybar(:,i) = b_param(Y(:,i), r_sum(Y(:,i+1:n), ones(1, n-i)), biasA, biasB, biasC);
        end
        
        % 应用第二变换：s_decep 和 s_multi
        Ybarbar = Ybar;
        shiftA = 0.35;
        shiftB = 0.001;
        shiftC = 0.05;
        Ybarbar(:,1:k) = s_decep(Ybar(:,1:k), shiftA, shiftB, shiftC);
        biasA = 30;
        biasB = 95;
        biasC = 0.35;
        Ybarbar(:,k+1:n) = s_multi(Ybar(:,k+1:n), shiftA, shiftB, shiftC);
        
        % 应用第三变换：r_nonsep
        T = NaN * ones(noSols, M);
        uLoop = M - 1;
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_nonsep(Ybarbar(:, lBnd:uBnd), k/(M-1));
        end
        T(:,M) = r_nonsep(Ybarbar(:,k+1:k+l), l);
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(9, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_concave(X(:,1:uLoop));
        y = F';
    end
end

%% WFG10 函数生成器
function p = wfg10(p, dim)
    global k l M;
    p.name = 'WFG10';
    p.pd = dim;
    p.od = M;
    p.domain = [zeros(dim,1) 2*[1:dim]'];
    p.func = @evaluate;
    
    % WFG10 评价函数
    function y = evaluate(x)
        % 初始化
        [noSols, n, S, D, A, Y] = wfg_initialize(x, p.od, k, l, 10);

        % 应用第一变换：s_multi
        shiftA = 30;
        shiftB = 10;
        shiftC = 0.35;
        Ybar = s_multi(Y, shiftA, shiftB, shiftC);
        
        % 应用第二变换：r_sum
        T = NaN * ones(noSols, M);
        lLoop = k + 1;
        uLoop = M - 1;
        weights = ones(1, k/(M-1));
        for i = 1:uLoop
            lBnd = 1 + (i-1)*k/(M-1);
            uBnd = i*k/(M-1);
            T(:,i) = r_sum(Ybar(:, lBnd:uBnd), weights);
        end
        T(:,M) = r_sum(Ybar(:, lLoop:n), ones(1, n - lLoop + 1));
        
        % 应用退化常数
        X = T;
        for i = 1:M-1
            X(:,i) = max(T(:,i), A(10, i)) .* (T(:,i) - 0.5) + 0.5;
        end
        
        % 生成目标函数值
        F = rep(D * X(:,M), [1, M]) + rep(S, [noSols, 1]) .* h_convex(X(:,1:uLoop));
        y = F';
    end
end

%% ----------%% CEC2009 系列无约束。参考文献：[4]---------
% CEC2009 UF（Unconstrained Function）测试问题

%% UF1 函数生成器
% x和y为列向量，输入x必须在搜索空间内，可以是矩阵
function p = uf1(p, dim)
    p.name = 'uf1';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF1 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = (x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp1 = sum(tmp(3:2:dim,:));  % 奇数索引
        tmp2 = sum(tmp(2:2:dim,:));  % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp2 / size(2:2:dim,2);
        clear tmp;
    end
end

%% UF2 函数生成器
function p = uf2(p, dim)
    p.name = 'uf2';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0;
    p.func = @evaluate;
    
    % UF2 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        X1 = repmat(x(1,:), [dim-1,1]);
        A = 6 * pi * X1 + pi/dim * repmat((2:dim)', [1, num]);
        tmp = zeros(dim, num);    
        tmp(2:dim,:) = (x(2:dim,:) - 0.3 * X1 .* (X1 .* cos(4.0 * A) + 2.0) .* cos(A)).^2;
        tmp1 = sum(tmp(3:2:dim,:));  % 奇数索引
        tmp(2:dim,:) = (x(2:dim,:) - 0.3 * X1 .* (X1 .* cos(4.0 * A) + 2.0) .* sin(A)).^2;
        tmp2 = sum(tmp(2:2:dim,:));  % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp2 / size(2:2:dim,2);
        clear X1 A tmp;
    end
end

%% UF3 函数生成器
function p = uf3(p, dim)
    p.name = 'uf3';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % UF3 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - repmat(x(1,:), [dim-1,1]).^(0.5 + 1.5 * (repmat((2:dim)', [1, num]) - 2.0) / (dim - 2.0));
        tmp1 = sum(Y(2:dim,:).^2, 1);
        tmp2 = sum(cos(20.0 * pi * Y(2:dim,:) ./ sqrt(repmat((2:dim)', [1, num]))), 1);
        tmp11 = 4.0 * sum(tmp1(3:2:dim,:)) - 2.0 * prod(tmp2(3:2:dim,:)) + 2.0;  % 奇数索引
        tmp21 = 4.0 * sum(tmp1(2:2:dim,:)) - 2.0 * prod(tmp2(2:2:dim,:)) + 2.0;  % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp11 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp21 / size(2:2:dim,2);
        clear Y;
    end
end

%% UF4 函数生成器
function p = uf4(p, dim)
    p.name = 'uf4';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.domain(1,2) = 1; % 第一个决策变量上界
    p.func = @evaluate;
    
    % UF4 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(Y(3:2:dim,:).^2, 1);  % 奇数索引
        tmp2 = sum(Y(4:2:dim,:).^2, 1);  % 偶数索引
        index1 = Y(2,:) < (1.5 - 0.75 * sqrt(2.0));
        index2 = Y(2,:) >= (1.5 - 0.75 * sqrt(2.0));
        Y(2,index1) = abs(Y(2,index1));
        Y(2,index2) = 0.125 + (Y(2,index2) - 1.0).^2;
        y(1,:) = x(1,:) + tmp1;
        y(2,:) = (1.0 - x(1,:)).^2 + Y(2,:) + tmp2;
        t = x(2,:) - sin(6.0 * pi * x(1,:) + 2.0 * pi/dim) - 0.5 * x(1,:) + 0.25;
        y(1,:) = y(1,:);
        y(2,:) = y(2,:);
        c(1,:) = Y(1,:) + Y(2,:) - 1.0;
        clear Y;
    end
end

%% UF5 函数生成器
function p = uf5(p, dim)
    p.name = 'uf5';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF5 评价函数
    function y = evaluate(x)
        N = 10.0;
        E = 0.1;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        H = zeros(dim, num);
        H(2:dim,:) = 2.0 * Y(2:dim,:).^2 - cos(4.0 * pi * Y(2:dim,:)) + 1.0;
        tmp1 = sum(H(3:2:dim,:));  % 奇数索引
        tmp2 = sum(H(2:2:dim,:));  % 偶数索引
        tmp = (0.5/N + E) * abs(sin(2.0 * N * pi * x(1,:)));
        y(1,:) = x(1,:) + tmp + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + tmp + 2.0 * tmp2 / size(2:2:dim,2);
        clear Y H;
    end
end

%% UF6 函数生成器
function p = uf6(p, dim)
    p.name = 'uf6';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF6 评价函数
    function y = evaluate(x)
        N = 2.0;
        E = 0.1;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(Y(2:dim,:).^2, 1);
        tmp2 = sum(cos(20.0 * pi * Y(2:dim,:) ./ sqrt(repmat((2:dim)', [1, num]))), 1);
        tmp11 = 4.0 * sum(tmp1(3:2:dim,:)) - 2.0 * prod(tmp2(3:2:dim,:)) + 2.0;  % 奇数索引
        tmp21 = 4.0 * sum(tmp1(2:2:dim,:)) - 2.0 * prod(tmp2(2:2:dim,:)) + 2.0;  % 偶数索引
        tmp = max(0, (1.0/N + 2.0 * E) * sin(2.0 * N * pi * x(1,:)));
        y(1,:) = x(1,:) + tmp + 2.0 * tmp11 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + tmp + 2.0 * tmp21 / size(2:2:dim,2);
        clear Y tmp1 tmp2;
    end
end

%% UF7 函数生成器
function p = uf7(p, dim)
    p.name = 'uf7';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0; % 第一个决策变量下界
    p.func = @evaluate;
    
    % UF7 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = (x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp1 = sum(Y(3:2:dim,:));  % 奇数索引
        tmp2 = sum(Y(2:2:dim,:));  % 偶数索引
        tmp = (x(1,:)).^0.2;
        y(1,:) = tmp + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - tmp + 2.0 * tmp2 / size(2:2:dim,2);
        clear Y;
    end
end

%% UF8 函数生成器
function p = uf8(p, dim)
    p.name = 'uf8';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % UF8 评价函数
    function y = evaluate(x)
        N = 2.0;
        a = 4.0;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * abs(sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0))) - 1.0;
        clear Y;
    end
end

%% UF9 函数生成器
function p = uf9(p, dim)
    p.name = 'uf9';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % UF9 评价函数
    function y = evaluate(x)
        N = 2.0;
        a = 3.0;
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y;
    end
end

%% UF10 函数生成器
function p = uf10(p, dim)
    p.name = 'uf10';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % UF10 评价函数
    function y = evaluate(x)
        x = x';
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]));
        H = zeros(dim, num);
        H(3:dim,:) = 4.0 * Y(3:dim,:).^2 - cos(8.0 * pi * Y(3:dim,:)) + 1.0;
        tmp1 = sum(H(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(H(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(H(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y H;
    end
end

%% ----------%% CEC2009 系列有约束。参考文献：[4]---------
% CEC2009 CF（Constrained Function）测试问题

%% CF1 函数生成器
% x和y为列向量，输入x必须在搜索空间内，可以是矩阵
function p = cf1(p, dim)
    p.name = 'cf1';
    p.pd = dim;
    p.od = 2;
    p.domain = [zeros(dim,1) ones(dim,1)];
    p.func = @evaluate;
    
    % CF1 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 10.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = (x(2:dim,:) - repmat(x(1,:), [dim-1,1]).^(0.5 + 1.5 * (repmat((2:dim)', [1, num]) - 2.0) / (dim - 2.0))).^2;
        tmp1 = sum(Y(3:2:dim,:));    % 奇数索引
        tmp2 = sum(Y(2:2:dim,:));    % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + 2.0 * tmp2 / size(2:2:dim,2);
        c(1,:) = y(1,:) + y(2,:) - a * abs(sin(N * pi * (y(1,:)-y(2,:)+1.0))) - 1.0;
        clear Y;
    end
end

%% CF2 函数生成器
function p = cf2(p, dim)
    p.name = 'cf2';
    p.pd = dim;
    p.od = 2;
    p.domain = [-1 * ones(dim,1) ones(dim,1)];
    p.domain(1,1) = 0;
    p.func = @evaluate;
    
    % CF2 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 2.0;
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = (x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp1 = sum(tmp(3:2:dim,:));    % 奇数索引
        tmp(2:dim,:) = (x(2:dim,:) - cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]))).^2;
        tmp2 = sum(tmp(2:2:dim,:));    % 偶数索引
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - sqrt(x(1,:)) + 2.0 * tmp2 / size(2:2:dim,2);
        t = y(2,:) + sqrt(y(1,:)) - a * sin(N * pi * (sqrt(y(1,:)) - y(2,:) + 1.0)) - 1.0;
        c(1,:) = sign(t) .* abs(t) ./ (1.0 + exp(4.0 * abs(t)));
        clear tmp;
    end
end

%% CF3 函数生成器
function p = cf3(p, dim)
    p.name = 'cf3';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.func = @evaluate;
    
    % CF3 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 2.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(2.0 * Y(3:2:dim,:).^2 - cos(4.0 * pi * Y(3:2:dim,:)) + 1.0, 1);  % 奇数索引
        tmp2 = sum(2.0 * Y(4:2:dim,:).^2 - cos(4.0 * pi * Y(4:2:dim,:)) + 1.0, 1);  % 偶数索引
        tmp = 0.5 * (1 - x(1,:)) - (1 - x(1,:)).^2;
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:).^2 + 2.0 * tmp2 / size(2:2:dim,2);
        c(1,:) = y(2,:) + y(1,:).^2 - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y;
    end
end

%% CF4 函数生成器
function p = cf4(p, dim)
    p.name = 'cf4';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF4 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(tmp(3:2:dim,:).^2, 1);  % 奇数索引
        tmp2 = sum(tmp(4:2:dim,:).^2, 1);  % 偶数索引
        index1 = tmp(2,:) < (1.5 - 0.75 * sqrt(2.0));
        index2 = tmp(2,:) >= (1.5 - 0.75 * sqrt(2.0));
        tmp(2,index1) = abs(tmp(2,index1));
        tmp(2,index2) = 0.125 + (tmp(2,index2) - 1.0).^2;
        y(1,:) = x(1,:) + tmp1;
        y(2,:) = (1.0 - x(1,:)).^2 + tmp(2,:) + tmp2;
        t = x(2,:) - sin(6.0 * pi * x(1,:) + 2.0 * pi/dim) - 0.5 * x(1,:) + 0.25;
        c(1,:) = sign(t) .* abs(t) ./ (1.0 + exp(4.0 * abs(t)));
        clear tmp index1 index2;
    end
end

%% CF5 函数生成器
function p = cf5(p, dim)
    p.name = 'cf5';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF5 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - 0.8 * repmat(x(1,:), [dim-1,1]) .* cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(2.0 * tmp(3:2:dim,:).^2 - cos(4.0 * pi * tmp(3:2:dim,:)) + 1.0, 1);  % 奇数索引
        tmp(2:dim,:) = x(2:dim,:) - 0.8 * repmat(x(1,:), [dim-1,1]) .* sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));    
        tmp2 = sum(2.0 * tmp(4:2:dim,:).^2 - cos(4.0 * pi * tmp(4:2:dim,:)) + 1.0, 1);  % 偶数索引
        index1 = tmp(2,:) < (1.5 - 0.75 * sqrt(2.0));
        index2 = tmp(2,:) >= (1.5 - 0.75 * sqrt(2.0));
        tmp(2,index1) = abs(tmp(2,index1));
        tmp(2,index2) = 0.125 + (tmp(2,index2) - 1.0).^2;
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(3:2:dim,2);
        y(2,:) = 1.0 - x(1,:) + tmp(2,:) + 2.0 * tmp2 / size(2:2:dim,2);
        c(1,:) = x(2,:) - 0.8 * x(1,:) .* sin(6.0 * pi * x(1,:) + 2.0 * pi/dim) - 0.5 * x(1,:) + 0.25;
        clear tmp;
    end
end

%% CF6 函数生成器
function p = cf6(p, dim)
    p.name = 'cf6';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF6 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - 0.8 * repmat(x(1,:), [dim-1,1]) .* cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(tmp(3:2:dim,:).^2, 1);  % 奇数索引
        tmp2 = sum(tmp(2:2:dim,:).^2, 1);  % 偶数索引
        y(1,:) = x(1,:) + tmp1;
        y(2,:) = (1.0 - x(1,:)).^2 + tmp2;
        tmp = 0.5 * (1 - x(1,:)) - (1 - x(1,:)).^2;
        c(1,:) = x(2,:) - 0.8 * x(1,:) .* sin(6.0 * pi * x(1,:) + 2 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));
        tmp = 0.25 * sqrt(1 - x(1,:)) - 0.5 * (1 - x(1,:));
        c(2,:) = x(4,:) - 0.8 * x(1,:) .* sin(6.0 * pi * x(1,:) + 4 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));    
        clear tmp;
    end
end

%% CF7 函数生成器
function p = cf7(p, dim)
    p.name = 'cf7';
    p.pd = dim;
    p.od = 2;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF7 评价函数
    function [y, c] = evaluate(x)
        x = x';
        [dim, num] = size(x);
        tmp = zeros(dim, num);
        tmp(2:dim,:) = x(2:dim,:) - cos(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp1 = sum(2.0 * tmp(3:2:dim,:).^2 - cos(4.0 * pi * tmp(3:2:dim,:)) + 1.0, 1);  % 奇数索引
        tmp(2:dim,:) = x(2:dim,:) - sin(6.0 * pi * repmat(x(1,:), [dim-1,1]) + pi/dim * repmat((2:dim)', [1, num]));
        tmp2 = sum(2.0 * tmp(6:2:dim,:).^2 - cos(4.0 * pi * tmp(6:2:dim,:)) + 1.0, 1);  % 偶数索引
        tmp(2,:) = tmp(2,:).^2;
        tmp(4,:) = tmp(4,:).^2;
        y(1,:) = x(1,:) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = (1.0 - x(1,:)).^2 + tmp(2,:) + tmp(4,:) + tmp2;
        tmp = 0.5 * (1 - x(1,:)) - (1 - x(1,:)).^2;
        c(1,:) = x(2,:) - sin(6.0 * pi * x(1,:) + 2 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));
        tmp = 0.25 * sqrt(1 - x(1,:)) - 0.5 * (1 - x(1,:));
        c(2,:) = x(4,:) - sin(6.0 * pi * x(1,:) + 4 * pi/dim) - sign(tmp) .* sqrt(abs(tmp));    
        clear tmp;
    end
end

%% CF8 函数生成器
function p = cf8(p, dim)
    p.name = 'cf8';
    p.pd = dim;
    p.od = 3;
    p.domain = [-4 * ones(dim,1) 4 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF8 评价函数
    function [y, c] = evaluate(x)
        x = x';
        N = 2.0;
        a = 4.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * abs(sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0))) - 1.0;
        clear Y;
    end
end

%% CF9 函数生成器
function p = cf9(p, dim)
    p.name = 'cf9';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF9 评价函数
    function [y, c] = evaluate(x)
        x = x';
        N = 2.0;
        a = 3.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        tmp1 = sum(Y(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(Y(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(Y(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y;
    end
end

%% CF10 函数生成器
function p = cf10(p, dim)
    p.name = 'cf10';
    p.pd = dim;
    p.od = 3;
    p.domain = [-2 * ones(dim,1) 2 * ones(dim,1)];
    p.domain(1:2,1) = 0; % 前两个决策变量下界
    p.domain(1:2,2) = 1; % 前两个决策变量上界
    p.func = @evaluate;
    
    % CF10 评价函数
    function [y, c] = evaluate(x)
        x = x';
        a = 1.0;
        N = 2.0;
        [dim, num] = size(x);
        Y = zeros(dim, num);
        Y(3:dim,:) = (x(3:dim,:) - 2.0 * repmat(x(2,:), [dim-2,1]) .* sin(2.0 * pi * repmat(x(1,:), [dim-2,1]) + pi/dim * repmat((3:dim)', [1, num]))).^2;
        H = zeros(dim, num);
        H(3:dim,:) = 4.0 * Y(3:dim,:).^2 - cos(8.0 * pi * Y(3:dim,:)) + 1.0;
        tmp1 = sum(H(4:3:dim,:));  % j-1 = 3*k
        tmp2 = sum(H(5:3:dim,:));  % j-2 = 3*k
        tmp3 = sum(H(3:3:dim,:));  % j-0 = 3*k
        y(1,:) = cos(0.5 * pi * x(1,:)) .* cos(0.5 * pi * x(2,:)) + 2.0 * tmp1 / size(4:3:dim,2);
        y(2,:) = cos(0.5 * pi * x(1,:)) .* sin(0.5 * pi * x(2,:)) + 2.0 * tmp2 / size(5:3:dim,2);
        y(3,:) = sin(0.5 * pi * x(1,:)) + 2.0 * tmp3 / size(3:3:dim,2);
        c(1,:) = (y(1,:).^2 + y(2,:).^2) ./ (1.0 - y(3,:).^2) - a * sin(N * pi * ((y(1,:).^2 - y(2,:).^2) ./ (1.0 - y(3,:).^2) + 1.0)) - 1.0;
        clear Y H;
    end
end

%% ------------------Transformation functions------------------
% ---------注意：以下函数不是测试问题，仅用于变换操作----------------

% WFG基准测试初始化
function [noSols, n, S, D, A, Y] = wfg_initialize(Z, M, k, l, testNo)
    % 检查输入参数数量
    if nargin ~= 5
        error('需要五个输入参数。');
    end

    % 获取决策变量数量和候选解数量
    [noSols, n] = size(Z);

    % 数据输入检查
    if n ~= (k + l)
        error('决策变量数量与k + l不一致。');
    end
    if rem(k, M-1) ~= 0
        error('k必须能被M-1整除。');
    end
    if (testNo == 2 || testNo == 3) && (rem(l,2) ~= 0)
        error('对于WFG2和WFG3，l必须是2的倍数。');
    end

    % 初始化函数范围内的常数
    NO_TESTS = 10;
    S = NaN * ones(1, M);
    for i = 1:M
        S(i) = 2 * i;
    end
    D = 1;
    A = ones(NO_TESTS, M-1);
    A(3,2:M-1) = 0;

    % 将所有变量范围转换到 [0,1]
    x = Z;
    for i = 1:n
        Y(:,i) = Z(:,i) ./ (2 * i);
    end
end

% Reduction: 加权和
function ybar = r_sum(y, weights)
    [noSols, noY] = size(y);
    wgtMatrix = repmat(weights, [noSols, 1]);
    ybar = y .* wgtMatrix;
    ybar = sum(ybar, 2) ./ sum(wgtMatrix, 2);
end

% Reduction: 非可分
function y_bar = r_nonsep(y, A)
    [noSols, noY] = size(y);
    y_bar = 0;
    for j = 1:noY
        innerSum = 0;
        for k = 0:(A-2)
            innerSum = innerSum + abs(y(:,j) - y(:,1 + mod(j + k, noY)));
        end
        y_bar = y_bar + y(:,j) + innerSum;
    end
    y_bar = y_bar / ((noY/A) * ceil(A/2) * (1 + 2*A - 2*ceil(A/2)));
end

% Bias: 多项式
function y_bar = b_poly(y, alpha)
    y_bar = y.^alpha;
end

% Bias: 平坦区域
function y_bar = b_flat(y, A, B, C)
    [noSols, noY] = size(y);
    min1 = min(0, floor(y - B));
    min2 = min(0, floor(C - y));
    y_bar = A + min1 * A .* (B - y) / B - min2 * (1 - A) .* (y - C) / (1 - C);
    % 由于机器精度问题，强制y_bar >= 0
    y_bar = max(0, y_bar);
end

% Bias: 参数依赖
function ybar = b_param(y, uy, A, B, C)
    [noSols, noY] = size(y);
    v = A - (1 - 2 * uy) .* abs(floor(0.5 - uy) + A);
    v = repmat(v, [1 noY]);
    ybar = y.^(B + (C - B) * v);
end

% Shift: 线性
function ybar = s_linear(y, A)
    ybar = abs(y - A) ./ abs(floor(A - y) + A);
end

% Shift: 欺骗性
function ybar = s_decep(y, A, B, C)
    y1 = floor(y - A + B) * (1 - C + (A - B)/B) / (A - B);
    y2 = floor(A + B - y) * (1 - C + (1 - A - B)/B) / (1 - A - B);
    ybar = 1 + (abs(y - A) - B) .* (y1 + y2 + 1/B);
end

% Shift: 多模态
function ybar = s_multi(y, A, B, C)
    y1 = abs(y - C) ./ (2 * (floor(C - y) + C));
    ybar = (1 + cos((4 * A + 2) * pi * (0.5 - y1)) + 4 * B * y1.^2) / (B + 2);
end

% Shape函数：线性
function f = h_linear(x)
    [noSols, mMinusOne] = size(x);
    M = mMinusOne + 1;
    f = NaN * ones(noSols, M);
    
    f(:,1) = prod(x, 2);
    for i = 2:mMinusOne
        f(:,i) = prod(x(:,1:M-i), 2) .* (1 - x(:,M-i+1));
    end
    f(:,M) = 1 - x(:,1);
end

% Shape函数：凸形
function f = h_convex(x)
    [noSols, mMinusOne] = size(x);
    M = mMinusOne + 1;
    f = NaN * ones(noSols, M);
    
    f(:,1) = prod(1 - cos(x * pi / 2), 2);
    for i = 2:mMinusOne
        f(:,i) = prod(1 - cos(x(:,1:M-i) * pi / 2), 2) .* (1 - sin(x(:,M-i+1) * pi / 2));
    end
    f(:,M) = 1 - sin(x(:,1) * pi / 2);
end

% Shape函数：凹形
function f = h_concave(x)
    [noSols, mMinusOne] = size(x);
    M = mMinusOne + 1;
    f = NaN * ones(noSols, M);
    
    f(:,1) = prod(sin(x * pi / 2), 2);
    for i = 2:mMinusOne
        f(:,i) = prod(sin(x(:,1:M-i) * pi / 2), 2) .* cos(x(:,M-i+1) * pi / 2);
    end
    f(:,M) = cos(x(:,1) * pi / 2);
end

% Shape函数：混合
function f = h_mixed(x, alpha, A)
    f = (1 - x(:,1) - cos(2 * A * pi * x(:,1) + pi/2) / (2 * A * pi)).^alpha;
end

% Shape函数：离散
function f = h_disc(x, alpha, beta, A)
    f = 1 - x(:,1).^alpha .* cos(A * x(:,1).^beta * pi).^2;
end

% 矩阵复制函数
function MatOut = rep(MatIn, REPN)
    % 获取输入矩阵的大小
    [N_D, N_L] = size(MatIn);
    
    % 计算复制索引
    Ind_D = rem(0:REPN(1)*N_D-1, N_D) + 1;
    Ind_L = rem(0:REPN(2)*N_L-1, N_L) + 1;
    
    % 创建输出矩阵
    MatOut = MatIn(Ind_D, Ind_L);
end
