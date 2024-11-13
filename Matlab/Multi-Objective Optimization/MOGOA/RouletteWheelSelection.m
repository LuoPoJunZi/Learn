% ---------------------------------------------------------
% 轮盘赌选择算法。通过一组权重，表示每个个体被选择的概率，
% 返回选择的个体的索引。
% 使用示例：
% fortune_wheel ([1 5 3 15 8 1])
%    最可能的结果是 4 （权重为 15）
% ---------------------------------------------------------

function choice = RouletteWheelSelection(weights)
  % 计算权重的累积和
  accumulation = cumsum(weights);
  
  % 生成一个 [0, accumulation(end)] 范围内的随机数
  p = rand() * accumulation(end);
  
  % 初始化选择的索引为 -1
  chosen_index = -1;
  
  % 遍历累积和，找到第一个大于随机数 p 的累积值，返回其索引
  for index = 1 : length(accumulation)
    if (accumulation(index) > p)
      chosen_index = index;
      break;
    end
  end
  
  % 返回选择的个体索引
  choice = chosen_index;
