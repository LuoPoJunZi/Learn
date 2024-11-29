在 MATLAB 中，可以使用 `parfor` 循环来实现并行计算，尤其适用于循环体中每次迭代是独立的情况。下面是一个简单的例子，演示如何使用并行计算加速大规模的计算任务。

假设我们要计算一系列矩阵的特征值，这个任务是可以独立并行完成的。代码如下：

```matlab
% 启动并行池（如果还未启动）
if isempty(gcp('nocreate'))
    parpool; % 默认开启并行池
end

% 矩阵数量
numMatrices = 100;
matrixSize = 500; % 每个矩阵的大小

% 随机生成一些矩阵
matrices = rand(matrixSize, matrixSize, numMatrices);

% 用于存储结果的变量
eigenvalues = cell(1, numMatrices);

% 并行计算特征值
tic;
parfor i = 1:numMatrices
    eigenvalues{i} = eig(matrices(:, :, i));
end
toc;

% 关闭并行池（可选）
% delete(gcp('nocreate'));

disp('特征值计算完成');
```

### 代码说明

1. **启动并行池**：`parpool` 命令用于开启并行池，通常 MATLAB 会自动检测并选择合适的线程数。也可以手动指定线程数，例如 `parpool(4)` 表示使用 4 个线程。
2. **生成矩阵**：随机生成一组 500x500 的矩阵。
3. **并行计算特征值**：使用 `parfor` 代替常规的 `for` 循环。`parfor` 循环中的每次迭代在不同的工作线程中独立执行，可以大幅缩短计算时间。
4. **关闭并行池**：完成后可以手动关闭并行池以节省资源（可选）。

### 运行结果

代码运行后会输出计算时间，并显示“特征值计算完成”。
