% output.m 文件
% 此脚本用于计算和输出模型参数的初始化值

% 定义决策变量 x 的具体值
x = [0.261853019189095, 0.954033395717644, 0.929755925150390, ...
     0.0420539659350538, 0.662361595432101, 0.150500955291545, ...
     0.948314964274653, 0.0110000915659857, 0.0856124488473688, ...
     0.918417984591724];

% 根据决策变量计算各个参数
T = 343 + x(1) * (363 - 343); % 计算温度 T，范围从 343 到 363
P = 2 + x(2) * (5 - 2);       % 计算压力 P，范围从 2 到 5
sigma_mem = 0.1 + x(3) * (0.3 - 0.1); % 计算膜的导电性 sigma_mem，范围从 0.1 到 0.3
tmem = 0.02 + x(4) * (0.05 - 0.02);   % 计算膜的厚度 tmem，范围从 0.02 到 0.05
dp = 1.00E-07 + x(5) * (6.00E-07 - 1.00E-07); % 计算孔的平均半径 dp，范围从 1E-7 到 6E-7
dm = 0.0001 + x(6) * (0.0005 - 0.0001); % 计算膜的厚度 dm，范围从 0.0001 到 0.0005
eps = 0.5 + x(7) * (0.9 - 0.5); % 计算膜的孔隙率 eps，范围从 0.5 到 0.9
da = 0.001 + x(8) * (0.006 - 0.001); % 计算气隙的厚度 da，范围从 0.001 到 0.006
Tc = 300 + x(9) * (308 - 300); % 计算冷却液的温度 Tc，范围从 300 到 308
c3 = 0.002; % 设定常数 c3
j = 0.2 + x(10) * (2 - 0.2); % 计算电流密度 j，范围从 0.2 到 2

% 以上变量的计算根据输入的决策变量 x，得到具体的模型参数
