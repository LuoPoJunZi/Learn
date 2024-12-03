function [x, endPop, bPop, traceInfo] = ga(bounds, evalFN, evalOps, startPop, opts, ...
termFN, termOps, selectFN, selectOps, xOverFNs, xOverOps, mutFNs, mutOps)
% ga - 遗传算法的主函数，用于优化问题
%
% 输出参数:
%   x         - 在优化过程中找到的最佳解
%   endPop    - 最终种群
%   bPop      - 最佳种群的跟踪记录
%   traceInfo - 每代的最佳和平均适应度信息
%
% 输入参数:
%   bounds     - 优化变量的上下界矩阵
%   evalFN     - 评估函数的名称（通常是一个 .m 文件）
%   evalOps    - 传递给评估函数的选项（默认为 []）
%   startPop   - 初始种群矩阵
%   opts       - [epsilon prob_ops display] 
%                epsilon：考虑两个适应度不同所需的最小差异
%                prob_ops：如果为 0，则以概率应用遗传操作；为 1 则使用确定性的操作应用次数
%                display：是否显示进度（1 为显示，0 为静默）
%   termFN     - 终止函数的名称（默认为 'maxGenTerm'）
%   termOps    - 传递给终止函数的选项（默认为 100）
%   selectFN   - 选择函数的名称（默认为 'normGeomSelect'）
%   selectOps  - 传递给选择函数的选项（默认为 0.08）
%   xOverFNs   - 交叉函数的名称字符串（空格分隔）
%   xOverOps   - 传递给交叉函数的选项矩阵
%   mutFNs     - 变异函数的名称字符串（空格分隔）
%   mutOps     - 传递给变异函数的选项矩阵
%
% 示例:
%   [bestSol, finalPop, bestPopTrace, traceInfo] = ga(bounds, 'gabpEval', [], initPop, [1e-6, 1, 1], ...
%                                'maxGenTerm', 100, 'normGeomSelect', 0.08, 'arithXover', 2, ...
%                                'nonUnifMutation', [2, 50, 3]);

    %% 初始化参数
    n = nargin;
    if n < 2 || n == 6 || n == 10 || n == 12
      disp('参数不足'); 
    end
    
    % 默认评估选项
    if n < 3 
      evalOps = [];
    end
    
    % 默认参数
    if n < 5
      opts = [1e-6, 1, 0];
    end
    
    % 默认参数
    if isempty(opts)
      opts = [1e-6, 1, 0];
    end
    
    %% 判断是否为 M 文件
    if any(evalFN < 48) % 判断 evalFN 是否包含非字符（ASCII 码小于 48 的字符）
      % 浮点数编码 
      if opts(2) == 1
        e1str = ['x=c1; c1(xZomeLength)=', evalFN ';'];  
        e2str = ['x=c2; c2(xZomeLength)=', evalFN ';']; 
      % 二进制编码
      else
        e1str = ['x=b2f(endPop(j,:),bounds,bits); endPop(j,xZomeLength)=', evalFN ';'];
      end
    else
      % 浮点数编码
      if opts(2) == 1
        e1str = ['[c1 c1(xZomeLength)]=' evalFN '(c1,[gen evalOps]);'];  
        e2str = ['[c2 c2(xZomeLength)]=' evalFN '(c2,[gen evalOps]);'];
      % 二进制编码
      else
        e1str=['x=b2f(endPop(j,:),bounds,bits);[x v]=' evalFN ...
    	'(x,[gen evalOps]); endPop(j,:)=[f2b(x,bounds,bits) v];'];  
      end
    end
    
    %% 默认终止信息
    if n < 6
      termOps = 100;
      termFN = 'maxGenTerm';
    end
    
    %% 默认变异信息
    if n < 12
      % 浮点数编码
      if opts(2) == 1
        mutFNs = 'boundaryMutation multiNonUnifMutation nonUnifMutation unifMutation';
        mutOps = [4, 0, 0; 6, termOps(1), 3; 4, termOps(1), 3;4, 0, 0];
      % 二进制编码
      else
        mutFNs = 'binaryMutation';
        mutOps = 0.05;
      end
    end
    
    %% 默认交叉信息
    if n < 10
      % 浮点数编码
      if opts(2) == 1
        xOverFNs = 'arithXover heuristicXover simpleXover';
        xOverOps = [2, 0; 2, 3; 2, 0];
      % 二进制编码
      else
        xOverFNs = 'simpleXover';
        xOverOps = 0.6;
      end
    end
    
    %% 仅默认选择选项，即轮盘赌。
    if n < 9
      selectOps = [];
    end
    
    %% 默认选择信息
    if n < 8
      selectFN = 'normGeomSelect';
      selectOps = 0.08;
    end
    
    %% 默认终止信息
    if n < 6
      termOps = 100;
      termFN = 'maxGenTerm';
    end
    
    %% 没有指定的初始种群
    if n < 4
      startPop = [];
    end
    
    %% 随机生成种群
    if isempty(startPop)
      startPop = initializega(80, bounds, evalFN, evalOps, opts(1: 2));
    end
    
    %% 二进制编码
    if opts(2) == 0
      bits = calcbits(bounds, opts(1));
    end
    
    %% 参数设置
    xOverFNs     = parse(xOverFNs); % 解析交叉函数名称字符串
    mutFNs       = parse(mutFNs);   % 解析变异函数名称字符串
    xZomeLength  = size(startPop, 2); 	          % xzome 的长度，即变量数 + 适应度
    numVar       = xZomeLength - 1; 	          % 变量数
    popSize      = size(startPop,1); 	          % 种群人口个数
    endPop       = zeros(popSize, xZomeLength);   % 初始化下一代种群矩阵
    numXOvers    = size(xOverFNs, 1);             % 交叉算子的数量
    numMuts      = size(mutFNs, 1); 		      % 变异算子的数量
    epsilon      = opts(1);                       % 两个适应度值被认为不同的阈值
    oval         = max(startPop(:, xZomeLength)); % 当前种群的最佳适应度值
    bFoundIn     = 1; 			                  % 记录最佳解变化的次数
    done         = 0;                             % 标志是否完成遗传算法的演化
    gen          = 1; 			                  % 当前代数
    collectTrace = (nargout > 3); 		          % 是否收集每代的跟踪信息
    floatGA      = opts(2) == 1;                  % 是否使用浮点数编码
    display      = opts(3);                       % 是否显示进度

    %% 精英模型
    while(~done)
        %% 获取当前种群的最佳个体
        [bval, bindx] = max(startPop(:, xZomeLength));            % 当前种群的最佳适应度值及其索引
        best =  startPop(bindx, :);                              % 当前最佳个体
        if collectTrace
            traceInfo(gen, 1) = gen; 		                        % 当前代数
            traceInfo(gen, 2) = startPop(bindx,  xZomeLength);      % 当前代的最佳适应度值
            traceInfo(gen, 3) = mean(startPop(:, xZomeLength));     % 当前代的平均适应度值
            traceInfo(gen, 4) = std(startPop(:,  xZomeLength));    % 当前代的适应度标准差
        end
        
        %% 判断是否更新最佳解
        if ( (abs(bval - oval) > epsilon) || (gen == 1))
            % 更新显示
            if display
                fprintf(1, '\n%d %f\n', gen, bval);          
            end
            
            % 更新种群矩阵
            if floatGA
                bPop(bFoundIn, :) = [gen, startPop(bindx, :)]; 
            else
                bPop(bFoundIn, :) = [gen, b2f(startPop(bindx, 1 : numVar), bounds, bits)...
                    startPop(bindx, xZomeLength)];
            end
            
            bFoundIn = bFoundIn + 1;                      % 更新最佳解变化次数
            oval = bval;                                  % 更新最佳适应度值
        else
            if display
                fprintf(1,'%d ',gen);	                      % 否则仅更新代数
            end
        end
        
        %% 选择种群
        endPop = feval(selectFN, startPop, [gen, selectOps]); % 使用选择函数选择新的种群
        
        %% 使用遗传算子
        if floatGA
            % 处理浮点数编码
            for i = 1 : numXOvers
                for j = 1 : xOverOps(i, 1)
                    a = round(rand * (popSize - 1) + 1); 	     % 随机选择一个父代
                    b = round(rand * (popSize - 1) + 1); 	     % 随机选择另一个父代
                    xN = deblank(xOverFNs(i, :)); 	         % 获取交叉函数名称
                    [c1, c2] = feval(xN, endPop(a, :), endPop(b, :), bounds, [gen, xOverOps(i, :)]);
                    
                    % 确保生成新的个体
                    if all(c1(1 : numVar) == endPop(a, 1 : numVar))
                        c1(xZomeLength) = endPop(a, xZomeLength);
                    elseif all(c1(1:numVar) == endPop(b, 1 : numVar))
                        c1(xZomeLength) = endPop(b, xZomeLength);
                    else
                        eval(e1str);
                    end
                    
                    if all(c2(1 : numVar) == endPop(a, 1 : numVar))
                        c2(xZomeLength) = endPop(a, xZomeLength);
                    elseif all(c2(1 : numVar) == endPop(b, 1 : numVar))
                        c2(xZomeLength) = endPop(b, xZomeLength);
                    else
                        eval(e2str);
                    end
                    
                    endPop(a, :) = c1; % 更新父代 a
                    endPop(b, :) = c2; % 更新父代 b
                end
            end
            
            for i = 1 : numMuts
                for j = 1 : mutOps(i, 1)
                    a = round(rand * (popSize - 1) + 1); % 随机选择一个个体进行变异
                    c1 = feval(deblank(mutFNs(i, :)), endPop(a, :), bounds, [gen, mutOps(i, :)]);
                    if all(c1(1 : numVar) == endPop(a, 1 : numVar))
                        c1(xZomeLength) = endPop(a, xZomeLength);
                    else
                        eval(e1str);
                    end
                    endPop(a, :) = c1; % 更新变异后的个体
                end
            end
        else
            % 处理二进制编码
            for i = 1 : numXOvers
                xN = deblank(xOverFNs(i, :)); % 获取交叉函数名称
                cp = find((rand(popSize, 1) < xOverOps(i, 1)) == 1); % 根据概率选择进行交叉的个体
                
                if rem(size(cp, 1), 2) 
                    cp = cp(1 : (size(cp, 1) - 1)); % 确保交叉个体为偶数
                end
                cp = reshape(cp, size(cp, 1) / 2, 2); % 重塑为成对个体
                
                for j = 1 : size(cp, 1)
                    a = cp(j, 1); 
                    b = cp(j, 2); 
                    [endPop(a, :), endPop(b, :)] = feval(xN, endPop(a, :), endPop(b, :), ...
                        bounds, [gen, xOverOps(i, :)]);
                end
            end
            
            for i = 1 : numMuts
                mN = deblank(mutFNs(i, :)); % 获取变异函数名称
                for j = 1 : popSize
                    endPop(j, :) = feval(mN, endPop(j, :), bounds, [gen, mutOps(i, :)]);
                    eval(e1str);
                end
            end
        end
        
        %% 更新记录
        gen = gen + 1; % 更新代数
        done = feval(termFN, [gen, termOps], bPop, endPop); % 判断是否满足终止条件
        startPop = endPop; 			                      % 将下一代作为当前种群
        [~, bindx] = min(startPop(:, xZomeLength));         % 找到当前种群中适应度最差的个体
        startPop(bindx, :) = best; 		                  % 将最优个体替换最差个体，保持精英
    end
    
    [bval, bindx] = max(startPop(:, xZomeLength)); % 获取最终种群中的最佳适应度值及其索引
    
    %% 显示结果
    if display 
        fprintf(1, '\n%d %f\n', gen, bval);	  % 打印最终代数和最佳适应度值
    end
    
    %% 二进制编码
    x = startPop(bindx, :); % 获取最佳个体
    if opts(2) == 0
        x = b2f(x, bounds, bits); % 将二进制编码转换为浮点数
        bPop(bFoundIn, :) = [gen, b2f(startPop(bindx, 1 : numVar), bounds, bits)...
            startPop(bindx, xZomeLength)];
    else
        bPop(bFoundIn, :) = [gen, startPop(bindx, :)];
    end
    
    %% 赋值
    if collectTrace
        traceInfo(gen, 1) = gen; 		                      % 当前迭代次数
        traceInfo(gen, 2) = startPop(bindx, xZomeLength);   % 当前代的最佳适应度值
        traceInfo(gen, 3) = mean(startPop(:, xZomeLength)); % 当前代的平均适应度值
    end
end
