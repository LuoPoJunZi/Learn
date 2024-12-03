function newPop = normGeomSelect(oldPop, options)
% normGeomSelect - 基于归一化几何分布的选择函数
%
% 输入参数:
%   oldPop  - 当前种群矩阵，每行表示一个个体，最后一列为适应度值
%   options - 选项向量 [当前代数 选择最佳的概率]
%
% 输出参数:
%   newPop - 新选择的种群矩阵

    %% 交叉选择排序
    q = options(2); 				    % 选择最佳的概率
    e = size(oldPop, 2); 			    % xZome 的长度，即变量数 + 适应度
    n = size(oldPop, 1);  		        % 种群数目
    newPop = zeros(n, e); 		        % 初始化新种群矩阵
    fit = zeros(n, 1); 		            % 初始化选择概率向量
    x = zeros(n,2); 			        % 初始化排名和索引的排序列表
    x(:, 1) = (n : -1 : 1)'; 	        % 设置排名，从 n 到 1
    [~, x(:, 2)] = sort(oldPop(:, e));  % 根据适应度值排序，获取索引
    
    %% 相关参数
    r = q / (1 - (1 - q) ^ n); 			            % 归一化分布常数
    fit(x(:, 2)) = r * (1 - q) .^ (x(:, 1) - 1); 	% 生成选择概率
    fit = cumsum(fit); 			                    % 计算累积概率
    
    %% 生成随机数并选择新种群
    rNums = sort(rand(n, 1)); 			            % 生成 n 个排序的随机数
    fitIn = 1;                                      % 初始化循环控制变量
    newIn = 1; 			                            % 初始化新种群索引
    while newIn <= n 				                % 循环直到选择完所有新个体
      if(rNums(newIn) < fit(fitIn)) 		
        newPop(newIn, :) = oldPop(fitIn, :); 	    % 根据累积概率选择个体
        newIn = newIn + 1; 			                % 选择下一个新个体
      else
        fitIn = fitIn + 1; 			                % 进入下一个累积概率区间
      end
    end
end
