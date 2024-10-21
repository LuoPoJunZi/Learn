% 生成一组从0到1的等间距点
t = linspace(0, 1);
% 计算简单多目标函数的值
F = simple_mult(t');
% 绘制目标函数的曲线
plot(t, F', 'LineWidth', 2)
hold on

% 绘制绿色虚线，表示目标函数的约束区域
plot([0, 0], [0, 8], 'g--');
plot([1, 1], [0, 8], 'g--');
% 在图中标记最小值位置
plot([0, 1], [1, 6], 'k.', 'MarkerSize', 15);
text(-0.25, 1.5, 'Minimum(f_1(x))')  % 标注 f1 的最小值位置
text(.75, 5.5, 'Minimum(f_2(x))')    % 标注 f2 的最小值位置
hold off

% 添加图例和标签
legend('f_1(x)', 'f_2(x)')
xlabel({'x'; 'Tradeoff region between the green lines'})

% 使用 fminbnd 找到第一个目标函数的最小值
k = 1;
[min1, minfn1] = fminbnd(@(x)pickindex(x, k), -1, 2);
% 使用 fminbnd 找到第二个目标函数的最小值
k = 2;
[min2, minfn2] = fminbnd(@(x)pickindex(x, k), -1, 2);

goal = [minfn1, minfn2];  % 目标值数组

nf = 2; % 目标函数数量
N = 500; % 用于绘图的点数量
onen = 1/N;  % 每个点的增量
x = zeros(N+1, 1);  % 初始化 x 值
f = zeros(N+1, nf);  % 初始化 f 值
fun = @simple_mult;  % 定义目标函数
x0 = 0.5;  % 初始值
options = optimoptions('fgoalattain', 'Display', 'off');  % 设定目标达成的优化选项

% 对于每个目标函数权重，从 0 到 1 进行循环
for r = 0:N
    t = onen * r; % 当前权重
    weight = [t, 1 - t];  % 权重数组
    % 使用目标达成法求解问题
    [x(r + 1, :), f(r + 1, :)] = fgoalattain(fun, x0, goal, weight, ...
        [], [], [], [], [], [], [], options);
end

% 绘制目标函数值的散点图
figure
plot(f(:, 1), f(:, 2), 'ko');

% 绘制平滑的曲线图
figure
x1 = f(:, 1);  % 第一个目标函数值
y1 = f(:, 2);  % 第二个目标函数值
x2 = linspace(min(x1), max(x1));  % 创建用于插值的 x 值
y2 = interp1(x1, y1, x2, 'spline');  % 使用样条插值平滑曲线
xlabel('f_1')  % x 轴标签
ylabel('f_2')  % y 轴标签
plot(x2, y2);  % 绘制平滑曲线

% 定义简单的多目标函数
function f = simple_mult(x)
    % f(:,1) = sqrt(1+x.^2);  % 第一个目标函数
    % f(:,2) = 4 + 2*sqrt(1+(x-1).^2);  % 第二个目标函数

    n = numel(x);  % 获取 x 的元素个数
    f1 = x(1);  % 第一个目标值
    g = 1 + 9/(n-1) * sum(x(2:end));  % 计算 g 值
    h = 1 - sqrt(f1 / g);  % 计算 h 值
    f2 = g * h;  % 计算第二个目标值
    f = [f1; f2];  % 将目标值组合成列向量返回
end

% 定义索引选择函数
function z = pickindex(x, k)
    z = simple_mult(x);  % 计算目标函数值
    z = z(k);  % 返回第 k 个目标函数值
end
