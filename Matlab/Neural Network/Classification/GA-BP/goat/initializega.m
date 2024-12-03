function pop = initializega(num, bounds, evalFN, evalOps, options)
% initializega - 初始化遗传算法的种群
%
% 输入参数:
%   num        - 种群规模，即需要创建的个体数量
%   bounds     - 变量的边界矩阵，每行表示一个变量的 [高界 低界]
%   evalFN     - 评估函数的名称（通常是一个 .m 文件）
%   evalOps    - 传递给评估函数的选项（默认为 []）
%   options    - 初始化选项，[类型 精度]
%                type: 1 表示浮点数编码，0 表示二进制编码
%                prec: 变量的精度（默认为 1e-6）
%
% 输出参数:
%   pop - 初始化的种群矩阵，每行表示一个个体，最后一列为适应度值

    %% 参数初始化
    if nargin < 5
      options = [1e-6, 1]; % 默认精度为 1e-6，浮点数编码
    end
    if nargin < 4
      evalOps = [];
    end
    
    %% 编码方式
    if any(evalFN < 48)    % 如果 evalFN 包含非字符（ASCII 码小于 48 的字符），假定为 M 文件
      if options(2) == 1   % 浮点数编码
        estr = ['x=pop(i,1); pop(i,xZomeLength)=', evalFN ';'];  
      else                 % 二进制编码
        estr = ['x=b2f(pop(i,:),bounds,bits); pop(i,xZomeLength)=', evalFN ';']; 
      end
    else                   % 非 M 文件
      if options(2) == 1   % 浮点数编码
        estr = ['[ pop(i,:) pop(i,xZomeLength)]=' evalFN '(pop(i,:),[0 evalOps]);']; 
      else                 % 二进制编码
        estr = ['x=b2f(pop(i,:),bounds,bits);[x v]=' evalFN ...
    	'(x,[0 evalOps]); pop(i,:)=[f2b(x,bounds,bits) v];'];  
      end
    end
    
    %% 参数设置 
    numVars = size(bounds, 1); 		           % 变量数
    rng     = (bounds(:, 2) - bounds(:, 1))';  % 变量范围
    
    %% 编码方式
    if options(2) == 1               % 浮点数编码
      xZomeLength = numVars + 1; 	 % xZome 的长度是变量数 + 适应度
      pop = zeros(num, xZomeLength); % 分配新种群矩阵
      % 随机生成变量值，范围在 [低界, 高界] 之间
      pop(:, 1 : numVars) = (ones(num, 1) * rng) .* (rand(num, numVars)) + ...
        (ones(num, 1) * bounds(:, 1)');
    else                             % 二进制编码
      bits = calcbits(bounds, options(1)); % 计算每个变量的二进制位数
      pop = round(rand(num, sum(bits) + 1)); % 随机生成二进制编码的种群，最后一列为适应度值
    end
    
    %% 运行评估函数
    for i = 1 : num
      eval(estr); % 对每个个体运行评估函数，计算适应度值
    end
end
