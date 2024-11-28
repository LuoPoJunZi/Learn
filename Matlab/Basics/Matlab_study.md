# Matlab基础
## [博客原文链接](https://blog.luopojunzi.com/p/matlab-study/)

## 1.1 **计算机使用**
```matlab
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% sqrt: 求平方根函数
% log10: 以10为底的对数函数
% i, j: 表示复数单位
% Inf: 表示无穷大，通常是由于除以零等操作导致的结果
% eps: 表示一个非常小的数，用于浮点数计算中的误差表示
% NaN: 表示“不是一个数值”，通常由无效的数学操作产生
% pi: 圆周率的常量

%%%%%%%%%%%% 可以在命令行窗口打出iskeyword %%%%%%%%%%%%
% iskeyword 用于查看 MATLAB 中的保留关键字，这些关键字不能作为变量名使用

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% 变量名与函数名冲突的示例
cos = 'This string.'; 
% cos 被定义为一个字符串变量，不再是余弦函数
cos(8); 
% 此处 cos(8) 指的是字符串中的第 8 个字符，而不是调用余弦函数
% 不建议使用内置函数名或关键字作为变量名
clear cos; 
% 清除变量 cos，使得 cos 恢复为余弦函数
cos(8); 
% 现在调用余弦函数，计算 8 的余弦值
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%%%%%%%%%%%% Format指令 %%%%%%%%%%%%
% format 控制输出显示格式
pi = 3.141;  
% 默认情况下 MATLAB 只显示小数点后约四位

format long
% 使用 format long 可以显示更多位的小数

% 常见格式化选项:
format short   % 显示四位小数
format long    % 显示更多位数的小数
format shortE  % 以科学计数法显示
format longE   % 以科学计数法显示更多位
format bank    % 显示两位小数
format hex     % 以十六进制显示数值
format rat     % 将数值转换为有理数分数表示

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%%%%%%%%%%%% 常用函数 %%%%%%%%%%%%
% ;(分号): 抑制命令的输出
% 方向键: 可以用方向键调出之前输入的命令
clc;  % 清除命令窗口中的显示内容
clear;  % 删除工作区中的所有变量
who;  % 显示当前工作区中的变量名称
whos;  % 显示当前工作区中的变量及其详细信息
```
## 1.2 **向量和矩阵**
```matlab
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Array（数组、向量和矩阵）
% 行向量:
>> a = [1 2 3 4]  % 定义行向量，元素以空格隔开
% 列向量:
>> b = [1;2;3;4]  % 定义列向量，元素以分号隔开

% 试试以下操作:
>> a*b   % 计算行向量与列向量的内积，返回一个标量（数值）
>> b*a   % 计算列向量与行向量的外积，返回一个 4x4 的矩阵

%%% 索引（indexing）%%%
A = [1 21 6; 5 17 9;31 2 7]; % 3x3 矩阵
>> A(1,2)  % 访问矩阵第一行第二列的元素，即21
>> A(1)=1; A(2)=5; A(3)=31; A(4)=21; % 按列顺序将A的值替换为指定值
>> A([1 3 5]) = [1 31 17] % 修改矩阵中第1,3,5个元素
>> A([1 3; 1 3]) = [1 31; 1 31]  % 在矩阵的(1,1),(1,3),(3,1),(3,3)位置赋值
>> A([1 3],[1 3]) = [1 6; 31 7]  % 选择第1、3行和第1、3列，并修改这些位置的值

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% 冒号运算符 %%%
A = [1:100];  % 生成一个从1到100，间隔为1的向量
B = [1:2:99]; % 生成一个从1到99，间隔为2的等差数列
A(3,:)       % 取矩阵A的第3行所有列
A(3,:) = []; % 删除矩阵A的第3行
str = 'a':2:'z'; % 生成从字符'a'到'z'，间隔为2的字符数组

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% 数组拼接 %%%
A = [1 2; 3 4];
B = [9 9; 9 9];
F = [A B];    % 横向拼接A和B，生成一个2x4的矩阵
F = [A;B];    % 纵向拼接A和B，生成一个4x2的矩阵

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% 数组操作 %%%
A = [1 2 3;4 5 6; 7 8 9];  % 定义矩阵A
B = [3 3 3;2 4 9; 1 3 1];  % 定义矩阵B
a = 2;  % 定义标量a

y1 = A+B;      % 矩阵加法
y2 = A*B;      % 矩阵乘法（矩阵的第一行与第二个矩阵的第一列相乘并累加）
y3 = A.*B;     % 矩阵点乘（逐元素相乘）
y4 = A/B;      % 矩阵除法，相当于 A * inv(B)
y5 = A./B;     % 矩阵点除（逐元素相除）
x1 = A + a;    % 矩阵每个元素加上标量a
x2 = A / a;    % 矩阵每个元素除以标量a
x3 = A./a;     % 与x2效果相同
x4 = A^a;      % 矩阵A的平方，即 A * A
x5 = A.^a;     % 矩阵每个元素的平方
C = A';        % 矩阵A的转置

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% 一些特殊矩阵 %%%
linspace(): 生成线性等间距的向量
eye(n): n*n的单位矩阵，对角线元素为1
zeros(n1,n2): 生成n1*n2的零矩阵
ones(n1,n2): 生成n1*n2的全1矩阵
diag(): 生成对角矩阵，输入对角线上的元素
rand(): 生成均匀分布的随机数矩阵

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
max(A): 返回矩阵每列中最大的元素
max(max(A)): 返回矩阵所有元素中的最大值
min(A): 返回矩阵每列中最小的元素
sum(A): 返回矩阵每列元素的和
mean(A): 返回矩阵每列元素的平均值

sort(A): 对矩阵每列的元素进行升序排序
sortrows(A): 根据矩阵的第一列进行排序，但保持每行元素的完整性
size(A): 返回矩阵的大小（行数和列数）
length(A): 返回矩阵中最大维度的长度（只适用于向量）
find(A==5): 返回矩阵中值等于5的元素的位置
```
## 2.1 **撰写**
```matlab
%% 1. MATLAB Script 循环绘图
for i = 1:10
    x = linspace(0,10,101);
    plot(x, sin(x + i));
    print(gcf, '-deps', strcat('plot', num2str(i), '.ps'));
end
% 注释：两个百分号可以分节，上面可以单独运行一个小节，不用运行全部程序。
% 在左侧数字上点击横线可以单独运行到该位置，便于调试。程序缩进问题可以全选代码，右键选择“智能缩进”或使用快捷键 Ctrl+I。
% ~=: 不等于，&&: 逻辑与，||: 逻辑或。

%% 2. If-Else 条件语句
a = 3;
if rem(a, 2) == 0    % rem 表示余数，a除以2的余数
    disp('a is even')   % display 输出
else
    disp('a is odd')
end
% elseif：用于第二个条件

%% 3. Switch 语句
input_num = 1;
switch input_num
    case -1
        disp('negative 1');
    case 0
        disp('zero');
    case 1
        disp('positive -1');
    otherwise
        disp('other value');
end

%% 4. While 循环语句
n = 1;
while prod(1:n) < 1e100   % prod(1:n) 表示 n 的阶乘，1e100 表示 1 * 10 的 100 次方
    n = n + 1;
end

% While 循环计算 1 到 999 的和
n = 1;
while n < 999
    n = n + 1;
    a = sum(1:n);  % 1 + 2 + ... + 999 的和
end

%% 5. For 循环语句
for n = 1:10
    a(n) = 2^n;
end
disp(a);

% For 循环计算 1 到 999 的和
a = 0;
for n = 1:999
    a = n + a;
end

%% 6. 预分配内存与性能优化
% 不预宣告的内存分配
tic    % 记录开始时间
for ii = 1:2000
    for jj = 1:2000
        A(ii, jj) = ii + jj;
    end
end
toc    % 记录结束时间并显示所用时间

% 预分配内存提高性能
tic    % 记录开始时间
A = zeros(2000, 2000);   % 预分配内存
for ii = 1:size(A, 1)
    for jj = 1:size(A, 2)
        A(ii, jj) = ii + jj;
    end
end
toc    % 记录结束时间并显示所用时间

%% 7. 其他技巧
clear all   % 清除所有变量
close all   % 关闭所有图形窗口
...         % 换行符，表示同一行未结束
% Ctrl + C  % 可用于中断正在运行的程序
```
## 2.2 **函数**
```matlab
%% Scripts VS Functions
% 功能（可以引用）
%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% %%% function的读写和应用 %%%
% >> edit(which('mean.m'))  % 打开 MATLAB 自带的函数 'mean.m' 文件，查看其命名规则和结构

% function x = freebody(x0,v0,t)    % 定义自由落体运动函数
% x = x0 + v0.*t + 1/2*9.8*t.*t;    % 自由落体运动的公式  % 使用点乘是为了逐元素相乘

% %%% freebody 引用 %%% 注意，定义的函数要保存为独立文件（如 freebody.m）
% freebody(0,0,10)   % 调用函数，输入初始位置 x0=0，初始速度 v0=0，时间 t=10

%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% %%% Function Handles %%%
f = @(x) exp(-2*x);   % 定义函数句柄，f(x) = exp(-2*x)
x = 0:0.1:2;          % 定义 x 的取值范围
plot(x, f(x))         % 绘制函数图像
```
## 3.1 **字符串、结构体、单元格数组**
```matlab
%% Variables：string，structure，cell(字符串，结构体，单元格)
042:Data access
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% double 变成 integer %%%    int: 整数型    uint：无符号整数型
% double：双精度浮点数
% single：单精度浮点数 

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% Character（char: 字元，string：字符，字符串）%%%
s1 = 'h';
whos 
uint16(s1)

s2 = 'H';
whos 
uint16(s2)

%%% string 字符串 %%%
s1 = 'Example';
s2 = 'String';

s3 = [s1 s2];
s4 = [s1; s2];  % 报错的原因是字符串的长度不同

%%% logical operation %%%
str = 'aardvark';
'a' == str;

str(str == 'a') = 'z';  % 把 str 中的 'a' 替换成 'z'

s1 = 'I like the letter E';   % 如何将其倒序输出，思考...

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% structure （结构体，异值data） %%%
student.name = 'John Doe';
student.id = 'jdo2@sfu.ca';
student.number = '301073268';
student.grade = [100, 75, 73; 95, 91, 85.5; 100, 98, 72];

student(2).name = 'Jane Smith';  % 第二个同学
student(2).id = 'jsm@sfu.ca';
student(2).number = '301073270';
student(2).grade = [95, 85, 80; 90, 88, 84; 100, 96, 90];

%%% nesting Structures %%%
A = struct('data', [3 4 7; 8 0 1], 'nest', ...
    struct('testnum', 'Test 1', ...
    'xdata', [4 2 8], 'ydata', [7 1 6]));
A(2).data = [9 3 2; 7 6 5];
A(2).nest.testnum = 'Test 2';
A(2).nest.xdata = [3 4 2];
A(2).nest.ydata = [5 0 9];
A.nest;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% cell Array (单元数组)%%%     使用 : {}
A(1,1) = {[1 4 3; 0 5 8; 7 2 9]};
A(1,2) = {'Anne Smith'};
A(2,1) = {3 + 7i};
A(2,2) = {-pi:pi:pi};
A;

%%% 另外一种方式，把大括号放到等式左侧
A{1,1} = [1 4 3; 0 5 8; 7 2 9];
A{1,2} = 'Anne Smith';
A{2,1} = 3 + 7i;
A{2,2} = -pi:pi:pi;
A;

%%% Accessing Cell Array 读取内容 %%% 
C = A{1,1};   % 读取单元格的具体内容
D = A(1,1);   % 显示 [3x3 double] 的单元格

%%% number 转换 cell %%%
a = magic(3);
b = num2cell(a);             % 每个单元变成 cell
c = mat2cell(a, [1 1 1], 3); % 将矩阵分块为 cell 数组，[1 1 1] 为行分块，3 为列分块

%%% cat() %%% 把两个 cell 接起来
A = [1 2; 3 4]; B = [5 6; 7 8];
C = cat(1, A, B);    % 行接起来，4x2 矩阵
C = cat(2, A, B);    % 列接起来，2x4 矩阵
C = cat(3, A, B);    % 层接起来，2x2x2 矩阵

%%% reshape() %%%
A = {'James Bond', [1 2; 3 4; 5 6]; pi, magic(5)};  % A 是一个 2x2 的 cell 数组
C = reshape(A, 1, 4);                               % C 变成一个 1x4 的 cell 数组
```
## 3.2 **储存数据**
```matlab
%% File Access (储存数据)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% load 和 save %%%  
%% save %%
clear;
a = magic(4);
save mydata1.mat          % MATLAB 格式保存
save mydata2.mat -ascii   % -ascii 可以用一般文字编辑器打开

%% load %%
load('mydata1.mat');               % 加载 MATLAB 文件
load('mydata2.mat', '-ascii');     % 加载 ASCII 文件

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% Excel File Reading: xlsread() %%%
Score = xlsread('04Score.xlsx');          % 读取整个表格数据
Score = xlsread('04Score.xlsx', 'B2:D4'); % 读取指定范围数据

% 计算每行的平均值，M 是均值列向量
M = mean(Score')';                       
xlswrite('04Score.xlsx', M, 1, 'E2:E4');  % 写入 Excel 文件
% 依次为：filename（文件名），variable（变量），sheet（表格编号），location（单元格位置）
xlswrite('04Score.xlsx', {'Mean'}, 1, 'E1');  % 写入标题

% 读取数据并获取表头信息
[Score, Header] = xlsread('04Score.xlsx');   % Header 是 cell 类型，包含表头

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% Low-level File Input/Output %%%
x = 0:pi/10:pi;    % 生成 x 的值
y = sin(x);        % 对应的 y = sin(x) 值
fid = fopen('sinx.txt', 'w');   % 打开文件以写入模式

for i = 1:11
    fprintf(fid, '%5.3f %8.4f\n', x(i), y(i));  % 格式化输出 x 和 y
    % %5.3f 总的宽度为 5，包含小数点后 3 位数字
    % %8.4f 总的宽度为 8，包含小数点后 4 位数字
    % \n 表示换行
end
fclose(fid);       % 关闭文件
type sinx.txt;     % 查看文件内容

%%% Read %%%
fid = fopen('sinx.txt', 'r');   % 打开文件以读取模式
A = fscanf(fid, '%f', [2 inf]);  % 按格式读取数据，读取两列，行数不限
fclose(fid);
A = A';  % 转置，以便数据行列符合原始文件格式
```
## 4. **初阶绘图**
```matlab
%% 初阶绘图
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%%% 基本绘图 %%%
x = 0:0.1:2*pi;
y = sin(x);
plot(x, y);      % 绘制 sin(x) 图像

hold on;         % 用于在同一图上绘制多个图像
plot(x, cos(x)); % 绘制 cos(x)
hold off;        % 停止绘制

%%% legend() 图例 %%%
x = 0:0.5:4*pi;
y = sin(x); h = cos(x); w = 1./(1 + exp(-x));
g = (1/(2*pi*2)^0.5).*exp(-1.*(x-2*pi).^2)./(2*2^2);
plot(x, y, 'bd-', x, h, 'gp:', x, w, 'ro-', x, g, 'c^-');
legend('sin(x)', 'cos(x)', 'logistic', 'gaussian'); % 添加图例
title('x-y 关系图');     % 图标题
xlabel('x');             % x 轴标签
ylabel('y');             % y 轴标签

%%% 添加标注和箭头 %%%
x = linspace(0, 3, 100);
y = x.^2.*sin(x);
plot(x, y);
line([2, 2], [0, 2^2*sin(2)]); % 垂直线

% 添加带 LaTeX 格式的数学公式
str = '$$ \int_{0}^{2} x^2\sin(x)\, dx $$';       
text(0.25, 2.5, str, 'Interpreter', 'latex');  % 公式标注
annotation('arrow', 'X', [0.32, 0.5], 'Y', [0.6, 0.4]);  % 添加箭头

%% 图像调整 (Figure Adjustment) %%%
x = linspace(0, 2*pi, 1000);
y = sin(x);
h = plot(x, y);

% 设置图的背景颜色为白色
set(gcf, 'Color', [1 1 1]);

% 设置坐标范围和字体
set(gca, 'XLim', [0, 2*pi]);          % X 轴范围
set(gca, 'YLim', [-1.2, 1.2]);        % Y 轴范围
set(gca, 'FontSize', 25);             % 字体大小
set(gca, 'XTick', 0:pi/2:2*pi);       % X 轴刻度
set(gca, 'XTickLabel', 0:90:360);     % 把 X 轴刻度改为角度

% 设置线条样式、宽度、颜色
set(h, 'LineStyle', '-.', 'LineWidth', 7.0, 'Color', 'g');

%% Marker Specification (标记样式) %%%
x = rand(20, 1);    % 随机生成数据
plot(x, '-md', 'LineWidth', 2, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', 'g', 'MarkerSize', 10);   % 绘制带标记的图
xlim([1, 20]);      % 设置 X 轴范围

%%% 绘制多个图像 (Subplots) %%%
x = -10:0.1:10;
y1 = x.^2 - 8;
y2 = exp(x);

subplot(2, 1, 1);       % 绘制第一个子图
plot(x, y1);
title('y = x^2 - 8');

subplot(2, 1, 2);       % 绘制第二个子图
plot(x, y2);
title('y = e^x');

%%% 保存图像 %%%
saveas(gcf, 'myfigure', 'png');   % 保存为 PNG 格式
print('-dpng', 'highres_figure.png');   % 高分辨率保存
```
## 5.1 **高阶绘图**
```matlab
%% 绘制基本图形（普通、半对数和全对数图）
x = logspace(-1, 1, 100); % 生成从 10^-1 到 10^1 的对数间隔数列，共 100 个点
y = x.^2;                 % 对应的 y 值为 x 的平方

% 使用 subplot 创建 2x2 的网格，在每个网格中绘制不同类型的图形
subplot(2,2,1);
plot(x, y);               % 普通的线性图
title('Plot');            % 设置图形标题

subplot(2,2,2);
semilogx(x, y);           % x 轴为对数尺度，y 轴为线性尺度的图
title('Semilogx');        % 设置图形标题

subplot(2,2,3);
semilogy(x, y);           % y 轴为对数尺度，x 轴为线性尺度的图
title('Semilogy');        % 设置图形标题

subplot(2,2,4);
loglog(x, y);             % x 和 y 轴都为对数尺度的图
title('Loglog');          % 设置图形标题


%% 双 Y 轴图（左侧和右侧不同数据）
x = 0:0.01:20;                        % x 轴数据从 0 到 20，步长为 0.01
y1 = 200 * exp(-0.05*x) .* sin(x);    % 左侧 Y 轴对应的函数，指数衰减的正弦函数
y2 = 0.8 * exp(-0.5*x) .* sin(10*x);  % 右侧 Y 轴对应的函数，快速衰减和快速振荡的正弦函数

[AX, H1, H2] = plotyy(x, y1, x, y2);  % 使用 plotyy 函数创建双 Y 轴图，AX 是两个 Y 轴的句柄
set(get(AX(1), 'YLabel'), 'String', 'Left Y-axis');  % 设置左侧 Y 轴标签
set(get(AX(2), 'YLabel'), 'String', 'Right Y-axis'); % 设置右侧 Y 轴标签
title('Labeling plotyy');             % 设置图形标题
set(H1, 'LineStyle', '--');           % 设置左侧数据线的样式为虚线
set(H2, 'LineStyle', ':');            % 设置右侧数据线的样式为点线


%% 直方图（Histogram，用于统计数据分布）
y = rand(1, 1000);        % 生成 1000 个 [0, 1] 范围内的随机数
subplot(2,1,1);
hist(y, 10);              % 将数据分成 10 个柱状绘制直方图
title('Bins = 10');       % 设置图形标题

subplot(2,1,2);
hist(y, 50);              % 将数据分成 50 个柱状绘制直方图，柱的数量越多，分布越细
title('Bins = 50');       % 设置图形标题


%% 条形图（Bar Charts，常用于展示类别数据）
x = [1 2 5 4 8];          % 条形图的第一个数据集
y = [x; 1:5];             % 第二个数据集，展示为矩阵

subplot(1,3,1);
bar(x);                   % 绘制一维数据集的条形图
title('A bargraph of vector x');  % 设置图形标题

subplot(1,3,2);
bar(y);                   % 绘制二维数据集的条形图，每个类别有两条数据
title('A bargraph of vector y');  % 设置图形标题

subplot(1,3,3);
bar3(y);                  % 绘制三维条形图，展示二维数据的高度
title('A 3D bargraph');   % 设置图形标题


%% 堆积条形图和水平条形图（Stacked and Horizontal Bar Charts）
subplot(1,2,1);
bar(y, 'stacked');        % 绘制堆积条形图，每个类别中的数据叠加在一起
title('Stacked');         % 设置图形标题

subplot(1,2,2);
barh(y);                  % 绘制水平条形图
title('Horizontal');      % 设置图形标题


%% 饼图（Pie Charts，用于展示比例关系）
a = [10 5 20 30];         % 各部分数据
subplot(1,3,1); 
pie(a);                   % 绘制基础饼图，显示比例

subplot(1,3,2); 
pie(a, [0,0,0,1]);        % 绘制带有分裂效果的饼图，最后一块突出显示
title('Pie chart with explosion'); % 设置图形标题

subplot(1,3,3); 
pie3(a, [0,0,0,1]);       % 绘制 3D 饼图，并带有分裂效果
title('3D Pie chart');     % 设置图形标题


%% 极坐标图（Polar Chart，用于角度和半径的关系展示）
x = 1:100; 
theta = x / 10;           % 角度 theta
r = log10(x);             % 半径 r 是 x 的对数

subplot(1,4,1); 
polar(theta, r);          % 绘制极坐标图，显示 r 与 theta 的关系

theta = linspace(0, 2*pi);  
r = cos(4*theta);         % 使用余弦函数创建新的极坐标数据
subplot(1,4,2); 
polar(theta, r);          % 绘制新的极坐标图

theta = linspace(0, 2*pi, 6);  % 创建六等分的角度（五边形）
r = ones(1, length(theta));    % 半径为 1 的固定值
subplot(1,4,3); 
polar(theta, r);               % 绘制五边形的极坐标图

theta = linspace(0, 2*pi);  
r = 1 - sin(theta);            % 创建不规则曲线的极坐标数据
subplot(1,4,4); 
polar(theta, r);               % 绘制新的极坐标图


%% 楼梯图和柱状图（Stairs and Stem）
x = linspace(0, 4*pi, 40);  % 创建从 0 到 4π 的 40 个等距点
y = sin(x);                 % 正弦函数

subplot(1,2,1);
stairs(x, y);               % 绘制楼梯状图，展示数据的阶梯变化
title('Stairs plot');       % 设置图形标题

subplot(1,2,2);
stem(x, y);                 % 绘制柱状图，展示数据的离散性
title('Stem plot');         % 设置图形标题

% 组合使用 plot 和 stem
t = linspace(0, 10);        % 创建从 0 到 10 的时间序列
y = sin(pi*t.^2);           % 计算正弦平方函数
plot(t, y, 'g');            % 绘制函数的绿色连线
hold on                     % 保留现有图形
stem(t, y, 'b');            % 叠加蓝色的柱状图
hold off                    % 释放 hold，使后续绘图不叠加


%% 箱线图和误差条（Boxplot and Error Bar）
load carsmall               % 加载示例数据集 carsmall，其中包含车辆 MPG 和 Origin 信息
boxplot(MPG, Origin);       % 绘制 MPG 根据 Origin 的箱线图，展示不同 Origin 下的 MPG 分布
title('Boxplot of MPG by Origin');  % 设置图形标题

x = 0:pi/10:pi;             % x 为从 0 到 pi 的等间隔值
y = sin(x);                 % 计算正弦函数
e = std(y) * ones(size(x)); % 计算每个点的标准差作为误差
errorbar(x, y, e);          % 绘制带误差条的正弦函数
title('Errorbar Plot');     % 设置图形标题
```
## 5.2 **填充**
```matlab
%% fill(填充)
% 绘制八边形停牌 (stop sign)
t = (1:2:15)'*pi/8; % 定义角度，用于生成正八边形的顶点
x = sin(t);         % 根据角度计算顶点的x坐标
y = cos(t);         % 根据角度计算顶点的y坐标
fill(x,y,'r');      % 使用fill函数绘制填充红色的多边形
axis square off;    % 设置坐标系为正方形并隐藏轴
text(0,0,'STOP','Color','w','FontSize',80,...   % 在中心添加文字“STOP”
    'FontWeight','bold','HorizontalAlignment','center');

%% Color Space（配色）
% 使用RGB颜色空间定义颜色
% [R G B]：分别表示红、绿、蓝通道，数值在0到1之间
% 例如：0表示最小强度（没有该颜色），1表示最大强度（颜色最强）
% 8-bit equivalence: 使用8位（0到255）表示
G = [46 38 29 24 13];   % 定义金牌数量
S = [29 27 17 26 8];    % 定义银牌数量
B = [29 23 19 32 7];    % 定义铜牌数量
h = bar(1:5,[G' S' B']); % 绘制柱状图表示5个国家在2012年奥运会中的奖牌数量
title('Medal count for top 5 countries in 2012 Olympics'); % 添加标题
ylabel('Number of medal'); % y轴标签
xlabel('Country'); % x轴标签
legend('Gold','Silver','Bronze'); % 图例

%% imagesc() - 图像显示
% 创建网格并计算z值
[x,y] = meshgrid(-3:.2:3,-3:.2:3);  % 创建二维网格
z = x.^2 + x.*y + y.^2;             % 计算每个点的z值
surf(x,y,z);                        % 生成三维表面图
box on;                             % 显示边框
set(gca,'FontSize',16);             % 设置坐标轴字体大小
zlabel('z');                        % z轴标签
xlim([-4 4]);   xlabel('x');        % x轴标签和范围
ylim([-4 4]);   ylabel('y');        % y轴标签和范围
imagesc(z);     axis square;        % 用色阶展示z矩阵并设置坐标轴为正方形
xlabel('x');    ylabel('y');        % 添加x和y轴标签
colorbar;                           % 显示色阶的颜色栏
colormap(hot);                      % 使用热色图（红色）
colormap(cool);                     % 使用冷色图（蓝色）
colormap(gray);                     % 使用灰度图（黑白）

%% 3D Plots - 三维绘图
x = 0:0.1:2*pi;
plot(x,sin(x));    % 绘制2D正弦曲线

% 使用plot3()绘制3D线
x = 0:0.1:3*pi;
z1 = sin(x);       % 第一条正弦曲线的z值
z2 = sin(2*x);     % 第二条正弦曲线的z值
z3 = sin(3*x);     % 第三条正弦曲线的z值
y1 = zeros(size(x));    % 第一条曲线的y坐标（在y=0处）
y3 = ones(size(x));     % 第三条曲线的y坐标（在y=1处）
y2 = y3./2;             % 第二条曲线的y坐标（在y=0.5处）
plot3(x,y1,z1,'r',x,y2,z2,'b',x,y3,z3,'g');  % 绘制三条三维曲线，红蓝绿三色
grid on;               % 开启网格线
xlabel('x-axis');   ylabel('y-axis');   zlabel('z-axis'); % 添加坐标轴标签

% meshgrid() 创建网格
x = -2:1:2;
y = -2:1:2;
[X,Y] = meshgrid(x,y); % 创建二维网格

% 使用mesh()和surf()绘制三维网格和表面
x = -3.5:0.2:3.5;
y = -3.5:0.2:3.5;
[X,Y] = meshgrid(x,y);    % 创建网格
Z = X.*exp(-X.^2 - Y.^2); % 计算Z值
subplot(1,2,1);     mesh(X,Y,Z);    % 绘制网格
subplot(1,2,2);     surf(X,Y,Z);    % 绘制带颜色的表面图

% 使用contour()绘制等高线图
subplot(1,2,1); mesh(X,Y,Z); axis square; % 绘制网格
subplot(1,2,2); contour(X,Y,Z); axis square; % 绘制等高线

% 使用contourf()填充等高线
subplot(1,3,1); mesh(X,Y,Z); contour(Z,[-0.45:.05:0.45]); % 添加特定等值线
subplot(1,3,2); [C,h] = contour(Z); clabel(C,h); % 显示等高线数值
subplot(1,3,3); contourf(Z); axis square; % 填充等高线

%% View Angle（视角）和光源
% 使用view()设定视角，使用light()为图形打光
[X,Y,Z] = sphere(64);       % 绘制球体
h = surf(X,Y,Z);            % 创建三维表面图
axis square vis3d off;      % 设置为正方形坐标轴并关闭轴
reds = zeros(256,3);        % 自定义颜色映射
reds(:,1) = (0:256-1)/255;
colormap(reds);             % 应用自定义红色渐变
shading interp;             % 使用插值阴影
lighting phong;             % 使用Phong光照模型
set(h,'AmbientStrength',0.75,'DiffuseStrength',0.5); % 设置光照强度
L1 = light('Position',[-1,-1,-1]); % 添加光源并设置位置和颜色
set(L1,'Position',[-1,-1,1]);   % 改变光源位置
set(L1,'Color','g');    % 设置光源颜色为绿色

%% patch() 绘制复杂多边形
v = [0 0 0;1 0 0;1 1 0;0 1 0;0.25 0.25 1;...  % 定义顶点坐标
    0.75 0.25 1;0.75 0.75 1;0.25 0.75 1];
f = [1 2 3 4; 5 6 7 8;1 2 6 5;2 3 7 6;...    % 定义多边形的面
    3 4 8 7;4 1 5 8];
subplot(1,2,1);     
patch('Vertices',v,'Faces',f,...    % 使用patch绘制平面多边形
    'FaceVertexCData',hsv(6),'FaceColor','flat');
view(3);    axis square tight;  grid on;   % 设置视角、正方形轴并显示网格

%% 小练习
% 加载cape数据，生成三维图形并应用自定义colormap
load cape
X = conv2(ones(9,9)/81,cumsum(cumsum(randn(100,100)),2)); % 生成数据
surf(X,'EdgeColor','none','EdgeLighting','Phong',...       % 绘制三维表面图
    'FaceColor','interp');
colormap(map);      caxis([-10,300]); % 应用colormap并设置颜色范围
grid off;       axis off;    % 关闭网格和坐标轴
```
## 6. **GUI**
`guide` 是 MATLAB 早期版本中的图形用户界面 (GUI) 设计工具，它允许用户以可视化方式创建应用程序界面（如按钮、菜单、图表等），并生成相应的 `.fig` 和 `.m` 文件来控制界面和响应事件。自 MATLAB R2016a 以来，`appdesigner` 被推荐为创建应用的工具，它整合了界面设计和代码管理，功能更为强大。

### 关于 `guide` 的操作步骤：
1. **启动 `guide`**  
   通过命令窗口输入 `guide`，会弹出一个 GUI 设计工具的窗口，用户可以从模板中选择现有布局或创建一个空白布局。

2. **设计界面**  
   在图形设计器中，拖放所需的 UI 元素（按钮、标签、输入框等）来构建界面。所有组件都会显示在 `figure` 窗口中。

3. **生成 `.fig` 和 `.m` 文件**  
   保存设计时，`guide` 会生成两个文件：
   - `.fig`：存储界面布局信息。
   - `.m`：包含响应事件的回调函数，用来控制界面行为。

4. **编辑代码**  
   在生成的 `.m` 文件中，可以编写对应的回调函数，定义按钮点击、输入等事件触发的操作。可以通过 MATLAB 编辑器修改 `.m` 文件来实现特定功能。

### App Designer 的替代功能：
1. **启动 App Designer**  
   通过命令 `appdesigner` 启动，它是创建和编辑 MATLAB 应用程序的现代工具，集成了 GUI 设计和代码开发环境。

2. **使用 App Designer**  
   - **图形设计**：与 `guide` 类似，可以拖放组件设计应用程序的用户界面。
   - **代码编辑**：App Designer 在界面中提供一个集成的代码窗口，可以同时查看和编辑 UI 组件的行为，代码和界面设计紧密结合。

3. **推荐使用 App Designer 的原因**：
   - 提供了更灵活的设计功能。
   - 支持对象属性和事件的更好管理。
   - 提供了更现代的 UI 组件。
   - 自动生成面向对象的代码，便于管理应用程序。

`guide` 已不再建议使用，转而推荐使用 `appdesigner`，尤其对于新项目或复杂应用开发来说，App Designer 更适合长期维护和扩展。
## 7. **图像处理**
```matlab
%%% Read and Show An Image（读取和显示影像）
% Read an image: imread()   读取图像文件
% Show an image: imshow()   显示图像
% Example:
clear, close all                % 清除工作区并关闭所有图形窗口
I = imread('pout.tif');         % 读取图像文件 'pout.tif'
imshow(I);                      % 显示读取的图像

%%% Image Processing（影像处理）
% immultiply() 影像的乘法操作，用来调整亮度
I = imread('rice.png');         % 读取图像文件 'rice.png'
subplot(1,2,1);                 % 创建一个包含 1 行 2 列的子图，激活第 1 个子图
imshow(I);                      % 显示原始图像
J = immultiply(I,1.5);          % 将图像亮度乘以 1.5（使图像变亮）
subplot(1,2,2);                 % 激活第 2 个子图
imshow(J);                      % 显示调整亮度后的图像

% imadd() 影像加法操作，将两幅图像相加
I = imread('rice.png');         % 读取第一张图像
J = imread('cameraman.tif');    % 读取第二张图像
K = imadd(I,J);                 % 对两幅图像进行加法操作，像素值相加
subplot(1,3,1);                 % 创建 1 行 3 列的子图，激活第 1 个子图
imshow(I);                      % 显示第一张图像
subplot(1,3,2);                 % 激活第 2 个子图
imshow(J);                      % 显示第二张图像
subplot(1,3,3);                 % 激活第 3 个子图
imshow(K);                      % 显示相加后的图像

% Image Histogram: imhist() 显示影像的直方图
imhist(I);                      % 显示图像 I 的直方图

% Histogram Equalization: histeq() 直方图均衡化
I = imread('pout.tif');         % 读取图像 'pout.tif'
I2 = histeq(I);                 % 对图像进行直方图均衡化处理
subplot(1,4,1);                 % 创建 1 行 4 列的子图，激活第 1 个子图
imhist(I);                      % 显示原始图像的直方图
subplot(1,4,2);                 % 激活第 2 个子图
imshow(I);                      % 显示原始图像
subplot(1,4,3);                 % 激活第 3 个子图
imshow(I2);                     % 显示均衡化后的图像
subplot(1,4,4);                 % 激活第 4 个子图
imhist(I2);                     % 显示均衡化后图像的直方图

%%% Geometric Transformation（几何变换）
% Image Rotation: imrotate() 旋转图像
I = imread('rice.png');         % 读取图像 'rice.png'
subplot(1,2,1);                 % 创建 1 行 2 列的子图，激活第 1 个子图
imshow(I);                      % 显示原始图像
J = imrotate(I,35,'bilinear');  % 将图像旋转 35 度，使用双线性插值法
subplot(1,2,2);                 % 激活第 2 个子图
imshow(J);                      % 显示旋转后的图像
size(I);                        % 显示原始图像的尺寸
size(J);                        % 显示旋转后图像的尺寸

% Write Image: imwrite() 保存影像
imwrite(I,'pout2.png');         % 将图像 I 保存为 'pout2.png'
```
### 解释：
- `imread()`：读取指定路径下的图像文件。
- `imshow()`：显示图像。
- `immultiply()`：对图像进行亮度调整（通过将图像像素值乘以一个系数）。
- `imadd()`：对两幅图像进行加法操作，将其像素值相加。
- `imhist()`：显示图像的灰度值直方图。
- `histeq()`：进行直方图均衡化，提高图像的对比度。
- `imrotate()`：对图像进行旋转操作。
- `imwrite()`：保存图像到指定路径。
## 8. **图像分割与处理**
```matlab
%%% Image Thresholding （图像二值化）
I = imread('rice.png');           % 读取图像 'rice.png'
imhist(I);                        % 显示图像的直方图
% graythresh() 找到自动阈值，im2bw() 将图像转换为二值图像
level = graythresh(I);            % 找到图像的灰度阈值
bw = im2bw(I,level);              % 将图像转换为二值图像（黑白）
subplot(1,2,1);                   % 创建 1 行 2 列的子图，激活第 1 个子图
imshow(I);                        % 显示原始图像
subplot(1,2,2);                   % 激活第 2 个子图
imshow(bw);                       % 显示二值图像

%%% Background Estimation（背景估计）
I = imread('rice.png');           % 读取图像 'rice.png'
BG = imopen(I,strel('disk',15));  % 使用形态学开运算，消除图像中小于 15 像素的物体
imshow(BG);                       % 显示背景图像

%%% Background Subtraction（背景消除）
I = imread('rice.png');           % 读取图像 'rice.png'
subplot(1,3,1);                   % 创建 1 行 3 列的子图，激活第 1 个子图
imshow(I);                        % 显示原始图像
BG = imopen(I,strel('disk',15));  % 估计背景
subplot(1,3,2);                   % 激活第 2 个子图
imshow(BG);                       % 显示背景
I2 = imsubtract(I,BG);            % 从原始图像中减去背景
subplot(1,3,3);                   % 激活第 3 个子图
imshow(I2);                       % 显示背景消除后的图像

%%% Thresholding on Background Removed Image （在背景移除图像上进行二值化）
I = imread('rice.png');           % 读取图像
level = graythresh(I);            % 找到图像的灰度阈值
bw = im2bw(I,level);              % 将原始图像转换为二值图像
subplot(1,2,1);                   % 创建 1 行 2 列的子图，激活第 1 个子图
imshow(bw);                       % 显示原始图像的二值化结果
BG = imopen(I,strel('disk',15));  % 背景估计
I2 = imsubtract(I,BG);            % 背景减除
level = graythresh(I2);           % 找到背景减除后的阈值
bw2 = im2bw(I2,level);            % 将减除背景后的图像进行二值化
subplot(1,2,2);                   % 激活第 2 个子图
imshow(bw2);                      % 显示减除背景后的二值化图像

%%% Connected-component Labeling （相连的像素标记）
% 使用 bwlabel() 进行相连区域标记
I = imread('rice.png');           % 读取图像 'rice.png'
BG = imopen(I,strel('disk',15));  % 背景估计
I2 = imsubtract(I,BG);            % 背景减除
level = graythresh(I2);           % 灰度阈值
BW = im2bw(I2,level);             % 二值化图像
[labeled,numObjects] = bwlabel(BW,8);  % 标记图像中相连的对象，8 表示使用 8 邻域

%%% Color-coding Objects（用颜色编码标记对象）
RGB_label = label2rgb(labeled);   % 用颜色编码相连对象的标签
imshow(RGB_label);                % 显示带有颜色标记的图像

%%% Object Properties: regionprops() （对象属性）
% 使用 regionprops() 提取图像中标记对象的属性
graindata = regionprops(labeled,'basic'); % 获取基础属性
graindata(51)                    % 显示第 51 个标记对象的属性

%%% Interactive Selection: bwselect() （交互式选择）
% 使用 bwselect() 手动选择图像中的对象
ObjI = bwselect(BW);             % 通过点击选择图像中的对象
imshow(ObjI);                    % 显示被选择的对象
```
### 解释：
- `graythresh()`：自动找到图像的灰度阈值，用于二值化。
- `im2bw()`：将图像转换为二值图像。
- `imopen()`：形态学开运算，用于去除噪声或小物体。
- `imsubtract()`：图像减法，通常用于背景减除。
- `bwlabel()`：标记二值图像中的相连对象。
- `label2rgb()`：为标记的对象赋予不同颜色。
- `regionprops()`：提取图像中标记对象的属性（如面积、周长等）。
- `bwselect()`：交互式选择图像中的对象。
## 9.1 **多项式微分和积分**
```matlab
%%% Polynomial Differentiation and Integration %%%

% Introduction to Polynomial Differentiation and Integration
% 多项式的微分和积分是微积分中的基本操作。在 MATLAB 中，可以使用多项式的系数向量来进行这些操作。
% 多项式一般表示为一个系数向量，例如：p = [a_n, a_(n-1), ..., a_1, a_0] 表示多项式
% f(x) = a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0。

% Polynomial Differentiation（多项式的微分）
% 微分是计算函数变化率的过程。对于多项式 f(x)，其导数 f'(x) 可以通过多项式的系数来计算。

% Polynomial Representation: 多项式表示方法（向量）
% Values of Polynomials: polyval() 用于计算多项式在给定点的值
a = [9, -5, 3, 7];            % 这是多项式 f(x) = 9*x^3 - 5*x^2 + 3*x + 7 的系数
x = -2:0.01:5;                % 定义 x 的范围，从 -2 到 5，每隔 0.01
f = polyval(a, x);            % 计算多项式在 x 处的值，返回对应的 f(x)
plot(x, f, 'LineWidth', 2);   % 绘制多项式图像，'LineWidth' 设置线条宽度
xlabel('x');                  % x 轴标签
ylabel('f(x)');               % y 轴标签
set(gca, 'FontSize', 14);     % 设置坐标轴字体大小，便于阅读

% Polynomial Differentiation: polyder() 计算多项式的导数
% 计算多项式的导数，得到一个新的多项式，该多项式的系数是原多项式的导数系数
p = [5, 0, -2, 0, 1];         % 这是多项式 f(x) = 5*x^4 - 2*x^2 + 1 的系数
dp = polyder(p);              % 计算导数，dp 是导数的系数向量
disp('Polynomial derivative coefficients:'); 
disp(dp);                     % 显示导数的系数
% 计算导数在 x = 7 时的值
value_at_x = polyval(dp, 7);  
disp(['Value of the derivative at x = 7: ', num2str(value_at_x)]);

% Polynomial Integration: polyint() 计算多项式的积分
% 积分是求多项式的累积量。对于多项式 f(x)，其积分是另一个多项式，加上一个常数项（积分常数）。

% 计算多项式的积分，得到一个新的多项式，其中包括积分常数
p = [5, 0, -2, 0, 1];         % 这是多项式 f(x) = 5*x^4 - 2*x^2 + 1 的系数
pi = polyint(p, 3);           % 计算积分，3 是积分常数
disp('Polynomial integral coefficients with constant term:');
disp(pi);                     % 显示积分的系数（包括常数项）
% 计算积分后的多项式在 x = 7 时的值
value_at_x_integrated = polyval(pi, 7);
disp(['Value of the integral at x = 7: ', num2str(value_at_x_integrated)]);
```
### 解释：
- `polyval(a, x)`：计算多项式 \( f(x) = a_1 x^{n-1} + a_2 x^{n-2} + \ldots + a_n \) 在给定 \( x \) 的值。`a` 是多项式的系数向量。
- `polyder(p)`：计算多项式的导数。`p` 是多项式的系数向量。
- `polyint(p, C)`：计算多项式的积分，其中 `C` 是积分常数。`p` 是多项式的系数向量。
- `polyval(dp, x)`：计算导数或积分后的多项式在给定 \( x \) 处的值。
## 9.2 **数值微分和积分**
```matlab
%%% Numerical Differentiation and Integration %%%

% Numerical Differentiation 数值微分
% 微分是计算函数变化率的过程。在离散数据点的情况下，我们可以通过计算相邻点之间的差异来估算导数。

% Differences: diff() 用于计算相邻数据点之间的差异
x = [1 2 5 2 1];  % 数据点
differences = diff(x);  % 计算相邻点之间的差值
disp('Differences between adjacent points:');
disp(differences);

% 计算两个点之间的斜率（差异），例如 (2, 7) 和 (1, 5)
x = [1 2];      % x 的值
y = [5 7];      % y 的值
slope = diff(y) ./ diff(x);  % 计算斜率（差异 / 差距）
disp('Slope between points (2, 7) and (1, 5):');
disp(slope);

% 练习：使用数值微分计算 sin 函数的导数
x0 = pi/2;   % 起始点
h = 0.1;      % 步长
x = [x0 x0 + h];  % 两个点
y = [sin(x0) sin(x0 + h)];  % 对应的 y 值
m = diff(y) ./ diff(x);  % 计算导数
disp('Numerical derivative of sin(x) at x = pi/2:');
disp(m);

% 使用更细的步长来计算 sin(x) 的导数
h = 0.5;    % 步长
x = 0:h:2*pi;  % x 的范围
y = sin(x);    % y = sin(x)
m = diff(y) ./ diff(x);  % 计算导数
plot(x(1:end-1), m, 'LineWidth', 2);  % 绘制导数图像
xlabel('x');
ylabel('f''(x)');
title('Numerical Derivative of sin(x)');

% 二次微分和三次微分
x = -2:0.005:2;  % x 的范围
y = x.^3;        % 多项式 f(x) = x^3
m = diff(y) ./ diff(x);       % 一次微分
m2 = diff(m) ./ diff(x(1:end-1)); % 二次微分，注意 x 的长度要减少 1
plot(x, y, 'b', x(1:end-1), m, 'r', x(1:end-2), m2, 'g');  % 绘制原函数及其微分
xlabel('x', 'FontSize', 18);
ylabel('y', 'FontSize', 18);
legend('f(x) = x^3', 'f''(x)', 'f''''(x)', 'Location', 'Best');
set(gca, 'FontSize', 18);

% Numerical Integration 数值积分
% 积分是计算函数在某个区间内的总量。可以使用不同的方法进行数值积分，如中点法、梯形法和辛普森法。

% Midpoint Rule Using sum() 使用矩形面积法计算积分
h = 0.05;      % 步长
x = 0:h:2;     % 积分区间
midpoint = (x(1:end-1) + x(2:end)) / 2;  % 计算每个小区间的中点
y = 4 * midpoint.^3;  % 被积函数
s = sum(h * y);       % 积分值（矩形面积的总和）
disp('Integral using midpoint rule:');
disp(s);

% Trapezoid Rule Using trapz() 使用梯形法计算积分
h = 0.05;      % 步长
x = 0:h:2;     % 积分区间
y = 4 * x.^3;  % 被积函数
s = trapz(x, y);  % 使用 trapz 函数计算积分
disp('Integral using trapezoid rule:');
disp(s);

% Second-order Rule Using Simpson's 1/3 Rule 辛普森法计算积分
h = 0.05;      % 步长
x = 0:h:2;     % 积分区间
y = 4 * x.^3;  % 被积函数
s = h / 3 * (y(1) + 2 * sum(y(3:2:end-2)) + 4 * sum(y(2:2:end)) + y(end)); 
disp('Integral using Simpson''s 1/3 rule:');
disp(s);

%%% Review of Function Handles @ %%%

% Numerical integration with integral() 使用积分函数进行积分
% Function handle 是一种引用函数的方式，允许将函数作为参数传递

% 定义函数句柄
y = @(x) 1 ./ (x.^3 - 2*x - 5);  % 被积函数的定义
% 使用 integral 函数计算积分
result = integral(y, 0, 2);  % 函数、积分下限、积分上限
disp('Integral using integral function:');
disp(result);

% Double and Triple Integration 双重积分和三重积分
% 双重积分：计算一个函数在二维区域内的积分
f = @(x, y) y .* sin(x) + x .* cos(y);  % 被积函数的定义
% 使用 integral2 函数计算双重积分
result2 = integral2(f, pi, 2*pi, 0, pi);  % 函数、x 和 y 的上下限
disp('Double integral using integral2 function:');
disp(result2);

% 三重积分：计算一个函数在三维区域内的积分
f = @(x, y, z) y .* sin(x) + z .* cos(y);  % 被积函数的定义
% 使用 integral3 函数计算三重积分
result3 = integral3(f, 0, pi, 0, 1, -1, 1);  % 函数、x、y 和 z 的上下限
disp('Triple integral using integral3 function:');
disp(result3);
```
### 解释：
1. **数值微分**：
   - **`diff()`**：用于计算相邻数据点之间的差值。例如，`diff(x)` 计算 `x` 向量中相邻元素的差值。
   - **斜率计算**：计算两个点之间的斜率，使用 `diff(y) ./ diff(x)`。
   - **导数计算**：对于连续的 x 值，我们可以使用 `diff(y) ./ diff(x)` 计算导数。

2. **数值积分**：
   - **中点法（Midpoint Rule）**：通过将每个区间的中点的函数值乘以区间宽度 `h` 来近似积分。
   - **梯形法（Trapezoid Rule）**：使用 `trapz()` 函数计算积分。它通过将每个区间的梯形面积累加来近似积分。
   - **辛普森法（Simpson's Rule）**：通过加权平均值来提高积分精度，尤其是对于非线性函数。

3. **函数句柄（Function Handles）**：
   - 使用 `@` 符号创建函数句柄，可以将函数作为参数传递给其他函数，例如 `integral()`、`integral2()` 和 `integral3()`。
## 10. **符号求解和数值根求解**
```matlab
%%% Symbolic Root Finding Approach 解析解（符号求解） %%%

% 定义符号变量
% 使用 syms 或 sym 定义符号变量
syms x              % 使用 syms 定义符号变量 x
x + x + x           % 这里的 x 是符号变量，计算结果为 3*x

(x + x + x) / 4     % 计算结果为 3*x/4

% 另一种定义符号变量的方式
x = sym('x');       % 使用 sym 定义符号变量 x
x + x + x           % 结果仍然是 3*x

(x + x + x) / 4     % 结果为 3*x/4

% 定义一个符号表达式
y = x^2 - 2*x - 8;  % 这里的 y 是一个关于 x 的符号表达式

% 使用 solve() 函数进行符号方程求解
syms x
y = x*sin(x) - x;   % 定义符号方程
solution = solve(y, x);  % 求解方程的根
disp('Symbolic solution of the equation x*sin(x) - x = 0:');
disp(solution);

% 解二元方程组
syms x y
eq1 = x - 2*y - 5;  % 定义第一个方程
eq2 = x + y - 6;    % 定义第二个方程
A = solve(eq1, eq2, x, y);  % 解方程组
disp('Solutions to the system of equations:');
disp(A);

% 解表达式中的未知数
syms x a b 
solve(a*x^2 - b)        % 默认求解 x
solve(a*x^2 - b, b)     % 求解 b 的值
disp('Symbolic solution for the equation a*x^2 - b = 0:');
disp(solve(a*x^2 - b));

% 符号微分
syms x
y = 4*x^5;         % 定义一个符号表达式
yprime = diff(y);  % 计算表达式的导数
disp('Symbolic derivative of 4*x^5:');
disp(yprime);

% 符号积分
syms x
y = x^2 * exp(x);  % 定义一个符号表达式
z = int(y);        % 计算不定积分
z = z - subs(z, x, 0); % 计算定积分（积分下限为 0）
disp('Symbolic integral of x^2 * exp(x):');
disp(z);

%%% Review of Function Handles（@） %%%

% 使用 fsolve() 求解非线性方程
f2 = @(x) (1.2*x + 0.3 + x*sin(x));  % 定义函数句柄
initial_guess = 0;    % 初始猜测值
solution = fsolve(f2, initial_guess);  % 使用 fsolve 计算方程的根
disp('Solution using fsolve:');
disp(solution);

% 使用 fzero() 求解非线性方程
f = @(x) x.^2;       % 定义一个函数句柄
root = fzero(f, 0.1);  % 使用 fzero 计算根
disp('Root using fzero:');
disp(root);

% 使用 fsolve() 和 fzero() 设置选项
options = optimset('MaxIter', 1e3, 'TolFun', 1e-10);  % 设置最大迭代次数和精度
solution_fsolve = fsolve(f, 0.1, options);  % 使用 fsolve 计算根
root_fzero = fzero(f, 0.1, options);  % 使用 fzero 计算根
disp('Solution using fsolve with options:');
disp(solution_fsolve);
disp('Root using fzero with options:');
disp(root_fzero);

% 多项式根的求解
% 定义一个多项式 f(x) = x^5 - 3.5*x^4 + 2.75*x^3 + 2.125*x^2 - 3.875*x + 1.25
p = [1 -3.5 2.75 2.125 -3.875 1.25];  % 多项式系数
roots_of_poly = roots(p);  % 使用 roots() 函数求解多项式的根
disp('Roots of the polynomial:');
disp(roots_of_poly);

% 二分法（Bracketing）和牛顿法（Newton-Raphson）介绍
% 二分法和牛顿法用于求解方程的根，可以通过编写特定的 MATLAB 代码实现
% 这里简要提及，具体代码参考相应算法的实现

% 递归函数示例
% 计算 N 的阶乘的递归函数
% function output = fact(n)
% % fact 递归计算 n!
% if n == 1
%     output = 1;
% else
%     output = n * fact(n-1);
% end
% end
% 这个函数计算 N 的阶乘，其中 fact 是递归调用的核心
```
### 解释：
1. **符号求解**：
   - **符号变量定义**：`syms` 和 `sym` 用于定义符号变量，便于进行符号计算。
   - **方程求解**：使用 `solve()` 函数求解符号方程，支持求解单个方程或方程组。
   - **符号微分和积分**：使用 `diff()` 和 `int()` 计算函数的导数和积分。

2. **函数句柄与数值求解**：
   - **`fsolve()`**：用于求解非线性方程，提供初始猜测值和选项参数。
   - **`fzero()`**：用于求解方程的零点，需要提供初始猜测值，并且函数必须穿过 x 轴。
   - **多项式根**：使用 `roots()` 函数求解多项式的根。

3. **二分法与牛顿法**：
   - 这些方法用于数值求解方程的根，具体实现可以参考相关的算法代码。

4. **递归函数**：
   - 示例展示了如何编写递归函数计算阶乘。

这些内容帮助初学者理解符号计算和数值求解的基本方法，并展示了 MATLAB 中相关功能的使用方法。
## 11. **线性方程**
```matlab
%%% Linear Equation 线性方程式 %%%

% Gaussian Elimination：rref() 高斯消去法
% 使用 rref() 函数将增广矩阵化为简化行最简形式
A = [1 2 1; 2 6 1; 1 1 4];  % 系数矩阵
b = [2; 7; 3];              % 常数项
augmented_matrix = [A b];   % 增广矩阵
R = rref(augmented_matrix); % 求解简化行最简形式
disp('Reduced Row Echelon Form (RREF) of the augmented matrix:');
disp(R);

% LU Factorization: lu() LU 分解
% LU 分解将矩阵分解为下三角矩阵 L 和上三角矩阵 U
A = [1 1 1; 2 3 5; 4 6 8];  % 系数矩阵
[L, U, P] = lu(A);          % LU 分解及置换矩阵 P
disp('L matrix:');
disp(L);
disp('U matrix:');
disp(U);
disp('P matrix (permutation matrix):');
disp(P);

% 解线性方程组 A*x = b
A = [1 2 1; 2 6 1; 1 1 4];  % 系数矩阵
b = [2; 7; 3];              % 常数项
x = A\b;                    % 解线性方程组的简洁方法（左除）
disp('Solution of the linear system A*x = b:');
disp(x);

% Cramer's (Inverse) Method 克莱默法则
% 通过计算矩阵的逆矩阵来求解线性方程组
A = [1 2 1; 2 6 1; 1 1 4];  % 系数矩阵
b = [2; 7; 3];              % 常数项
x = inv(A) * b;            % 计算解（注意：inv(A) 不一定存在）
disp('Solution using Cramer''s method (using matrix inverse):');
disp(x);

% Functions to Check Matrix Condition 矩阵条件数
% 检查矩阵的健康度，条件数高表示矩阵接近奇异
A = [1 2 3; 2 4.0001 6; 9 8 7]; % 例子矩阵
cond_A = cond(A);                % 计算条件数
disp('Condition number of matrix A:');
disp(cond_A);

B = [1 2 3; 2 5 6; 9 8 7]';     % 转置矩阵
cond_B = cond(B);                % 计算条件数
disp('Condition number of matrix B:');
disp(cond_B);

%%% Linear System 线性系统部分 %%%

% y = A * b
% 计算矩阵的特征向量和特征值
A = [2 -12; 1 -5];
[v, d] = eig(A);     % v 是特征向量矩阵，d 是对角特征值矩阵
disp('Eigenvectors (columns of matrix v):');
disp(v);
disp('Eigenvalues (diagonal elements of matrix d):');
disp(d);

% Matrix Exponential：expm() 矩阵指数
% 计算矩阵的指数，常用于微分系统
A = [0 -6 -1; 6 2 -16; -5 20 -10]; % 系数矩阵
x0 = [1 1 1]';                     % 初始条件
X = [];                            % 初始化结果矩阵
for t = 0:0.01:1
    X = [X expm(t*A) * x0];        % 计算矩阵指数并应用于初始条件
end
% 绘制三维图
plot3(X(1,:), X(2,:), X(3,:), '-o');
xlabel('x_1');
ylabel('x_2');
zlabel('x_3');
grid on;
axis tight square;
title('Matrix Exponential of A over Time');
```
### 解释：
1. **高斯消去法（Gaussian Elimination）**：
   - 使用 `rref()` 函数将增广矩阵化为简化行最简形式，从而求解线性方程组。

2. **LU 分解（LU Factorization）**：
   - 使用 `lu()` 函数将矩阵分解为下三角矩阵 `L` 和上三角矩阵 `U`，以及置换矩阵 `P`。LU 分解在解线性方程组和计算矩阵的行列式等方面非常有用。

3. **克莱默法则（Cramer's Method）**：
   - 使用 `inv()` 函数计算矩阵的逆矩阵来求解线性方程组。注意：矩阵的逆可能不存在，尤其是当矩阵接近奇异时。

4. **矩阵条件数（Matrix Condition Number）**：
   - 使用 `cond()` 函数计算矩阵的条件数，条件数高表示矩阵可能接近奇异，可能影响求解的稳定性。

5. **特征值和特征向量（Eigenvalues and Eigenvectors）**：
   - 使用 `eig()` 函数计算矩阵的特征值和特征向量，特征值和特征向量在许多线性代数问题中都非常重要。

6. **矩阵指数（Matrix Exponential）**：
   - 使用 `expm()` 函数计算矩阵的指数，用于解决涉及微分方程的线性系统。结果可视化为三维图形。

## 12. **统计功能及其应用**
```matlab
%%% Statistics 统计 %%%

% 1. Mean（平均值）、Median（中位数）、Mode（众数）、Quartile（四分位数）
% Max（最大值）、Min（最小值）
% std（标准差）: 标准差衡量数据的离散程度
% var（方差）: 方差是标准差的平方，衡量数据的离散程度的平方

x = 1:14;                 % 数据集
freqy = [1 0 1 0 4 0 1 0 3 1 0 0 1 1];  % 频数

% 绘制频数图
subplot(1,3,1);     % 将图像分为1行3列，当前绘制第1个子图
bar(x,freqy);       % 绘制条形图
xlim([0 15]);       % 设置x轴的显示范围

subplot(1,3,2);     % 当前绘制第2个子图
area(x,freqy);      % 绘制面积图
xlim([0 15]);       % 设置x轴的显示范围

subplot(1,3,3);     % 当前绘制第3个子图
stem(x,freqy);      % 绘制离散数据的垂直线图
xlim([0 15]);       % 设置x轴的显示范围

% 2. Boxplot（箱线图）
marks = [80 81 81 84 88 92 92 94 96 97]; % 学生成绩数据
boxplot(marks);                       % 绘制箱线图
title('Boxplot of Marks');            % 添加标题

% 计算四分位数
quartiles = prctile(marks, [25 50 75]); % 计算25%、50%、75%四分位数
disp('Quartiles:');
disp(quartiles);

% 3. Skewness（偏度）和Kurtosis（峰度）
% 偏度表示数据分布的对称性
% 峰度表示数据分布的尖峭程度

X = randn([10 3]) * 3;   % 生成随机数据，3列，每列数据乘以3
X(X(:,1)<0,1) = 0;      % 将第1列负值置为0，制造右偏分布
X(X(:,3)>0,3) = 0;      % 将第3列正值置为0，制造左偏分布

% 绘制箱线图比较不同偏度的分布
boxplot(X, {'Right-skewed','Symmetric','Left-skewed'});
title('Boxplot of Skewed Distributions'); % 添加标题

% 计算偏度
y = skewness(X);       % 计算数据的偏度
disp('Skewness:');
disp(y);

% 计算峰度
k = kurtosis(X);       % 计算数据的峰度
disp('Kurtosis:');
disp(k);

```

### 解释：

1. **频数图（Bar, Area, Stem）**：
   - `bar(x, freqy)`：绘制条形图，其中 `x` 表示数据点，`freqy` 表示每个数据点的频数。
   - `area(x, freqy)`：绘制面积图，以展示数据的分布。
   - `stem(x, freqy)`：绘制离散数据的垂直线图，用于显示每个数据点的频数。

2. **箱线图（Boxplot）**：
   - `boxplot(marks)`：绘制箱线图，展示数据的分布特征，包括中位数、四分位数和异常值。
   - `prctile(marks, [25 50 75])`：计算数据的25%、50%（中位数）、75%四分位数。

3. **偏度（Skewness）和峰度（Kurtosis）**：
   - `skewness(X)`：计算数据的偏度，描述数据分布的对称性。
   - `kurtosis(X)`：计算数据的峰度，描述数据分布的尖峭程度。偏度和峰度帮助分析数据的分布形态。

4. **绘制不同分布的箱线图**：
   - `boxplot(X, {'Right-skewed', 'Symmetric', 'Left-skewed'})`：展示三种不同偏度的分布，帮助理解偏度对数据分布的影响。

## 13. **简单线性回归与插值方法**

#### 1. 简单线性回归

**多项式曲线拟合（Polynomial Curve Fitting）**：
使用 `polyfit` 函数进行多项式拟合。例如，进行线性回归：

```matlab
x = [-1.2 -0.5 0.3 0.9 1.8 2.6 3.0 3.5];
y = [-15.6 -8.5 2.2 4.5 6.6 8.2 8.9 10.0];

fit = polyfit(x, y, 1);       % 返回线性回归系数 a 和 b
xfit = linspace(x(1), x(end), 100); % 拟合曲线的x范围
yfit = polyval(fit, xfit);    % 计算拟合值
plot(x, y, 'ro', xfit, yfit); % 绘制数据点和拟合线
set(gca, 'FontSize', 14);
legend('Data points', 'Best-fit');
```

**线性相关性分析**：
检验 `x` 和 `y` 是否存在线性关系：

```matlab
x = [-1.2 -0.5 0.3 0.9 1.8 2.6 3.0 3.5];
y = [-15.6 -8.5 2.2 4.5 6.6 8.2 8.9 10.0];

scatter(x, y);       % 绘制散点图
box on;              % 显示边框
axis square;         % 轴刻度相同
corrcoef(x, y)       % 计算相关系数
```

**高阶多项式回归**：
进行不同阶数的多项式回归：

```matlab
x = [-1.2 -0.5 0.3 0.9 1.8 2.6 3.0 3.5];
y = [-15.6 -8.5 2.2 4.5 6.6 8.2 8.9 10.0];
figure('Position', [50, 50, 1500, 400]);

for i = 1:3
    subplot(1, 3, i);
    p = polyfit(x, y, i);         % 拟合i阶多项式
    xfit = linspace(x(1), x(end), 100);
    yfit = polyval(p, xfit);      % 计算拟合值
    plot(x, y, 'ro', xfit, yfit);
    set(gca, 'FontSize', 14);
    ylim([-17, 11]);
    legend('Data points', 'Fitted curve');
end
```

**多变量回归**：
进行多元线性回归分析：

```matlab
load carsmall;
y = MPG;
x1 = Weight;
x2 = Horsepower;
X = [ones(length(x1), 1) x1 x2];
b = regress(y, X);              % 多元线性回归

% 绘制拟合平面
x1fit = min(x1):100:max(x1);
x2fit = min(x2):10:max(x2);
[X1FIT, X2FIT] = meshgrid(x1fit, x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
scatter3(x1, x2, y, 'filled');
hold on;
mesh(X1FIT, X2FIT, YFIT);
hold off;
xlabel('Weight');
ylabel('Horsepower');
zlabel('MPG');
view(50, 10);
```

#### 2. 内插与回归

**线性内插（Linear Interpolation）**：

```matlab
x = linspace(0, 2*pi, 40);
x_m = x;
x_m([11:13 28:30]) = NaN;
y_m = sin(x_m);

plot(x_m, y_m, 'ro', 'MarkerFaceColor', 'r');
xlim([0, 2*pi]);
ylim([-1.2, 1.2]);
box on;
set(gca, 'FontName', 'symbol', 'FontSize', 16);
set(gca, 'XTick', 0:pi/2:2*pi);
set(gca, 'XTickLabel', {'0', 'π/2', 'π', '3π/2', '2π'});

m_i = ~isnan(x_m);
y_i = interp1(x_m(m_i), y_m(m_i), x);
hold on;
plot(x, y_i, '-b', 'LineWidth', 2);
hold off;

% 平滑曲线内插（Spline Interpolation）
y_i = spline(x_m(m_i), y_m(m_i), x);
hold on;
plot(x, y_i, '-g', 'LineWidth', 2);
hold off;
h = legend('Original', 'Linear', 'Spline');
set(h, 'FontName', 'Times New Roman');
```

**立方样条与赫尔米特多项式**：

```matlab
x = -3:3;
y = [-1 -1 -1 0 1 1 1];
t = -3:.01:3;
s = spline(x, y, t);           % 立方样条插值
p = pchip(x, y, t);            % 赫尔米特插值

hold on;
plot(t, s, ':g', 'LineWidth', 2);
plot(t, p, '--b', 'LineWidth', 2);
plot(x, y, 'ro', 'MarkerFaceColor', 'r');
hold off;
box on;
set(gca, 'FontSize', 16);
h = legend('Original', 'Spline', 'Hermite');
```

**二维插值（interp2）**：

```matlab
xx = -2:.5:2;
yy = -2:.5:3;
[X, Y] = meshgrid(xx, yy);
Z = X .* exp(-X.^2 - Y.^2);

surf(X, Y, Z);
hold on;
plot3(X, Y, Z + 0.01, 'ok', 'MarkerFaceColor', 'r');

xx_i = -2:.1:2;
yy_i = -2:.1:3;
[X_i, Y_i] = meshgrid(xx_i, yy_i);
Z_i = interp2(xx, yy, Z, X_i, Y_i);
surf(X_i, Y_i, Z_i);
hold on;
plot3(X, Y, Z + 0.01, 'ok', 'MarkerFaceColor', 'r');

Z_i = interp2(xx, yy, Z, X_i, Y_i, 'cubic'); % 使用立方插值
surf(X_i, Y_i, Z_i);
hold on;
plot3(X, Y, Z + 0.01, 'ok', 'MarkerFaceColor', 'r');
```

### 总结

- **线性回归**：用 `polyfit` 和 `polyval` 进行多项式拟合。检查线性相关性使用 `corrcoef`。
- **插值**：用 `interp1` 进行一维内插，用 `spline` 进行平滑内插。对于二维数据，使用 `interp2` 进行内插，并比较不同插值方法（线性、立方等）的效果。

