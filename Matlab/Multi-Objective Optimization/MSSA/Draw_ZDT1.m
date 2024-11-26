% -------------------------------------------------------------
% 文件名: Draw_ZDT1.m
% 功能: 绘制ZDT1测试函数的目标函数前沿曲线
%       ZDT1是一个经典的多目标优化测试函数，具有两个目标函数。
% 输出:
%       TPF - 存储目标函数前沿曲线上的点（f1, f2）
% 作者: [您的名字]
% 日期: [日期]
% -------------------------------------------------------------

function TPF = Draw_ZDT1()
    % 定义目标函数句柄，调用ZDT1函数
    ObjectiveFunction = @(x) ZDT1(x);
    
    % 生成自变量x的取值范围，从0到1，步长为0.01
    x = 0:0.01:1;
    
    % 初始化存储目标函数前沿曲线点的矩阵
    % 每一行对应一个点的(f1, f2)值
    TPF = zeros(length(x), 2);
    
    % 遍历每一个x值，计算对应的f1和f2值
    for i = 1:length(x)
        % 假设决策变量只有两个，其中第二个变量固定为0
        TPF(i, :) = ObjectiveFunction([x(i), 0]);
    end
    
    % 绘制目标函数前沿曲线
    % 'LineWidth'设置线条宽度为2，增强可见性
    line(TPF(:, 1), TPF(:, 2), 'LineWidth', 2);
    
    % 设置图形标题
    title('ZDT1')
    
    % 设置x轴标签
    xlabel('f1')
    
    % 设置y轴标签
    ylabel('f2')
    
    % 显示坐标轴框线
    box on
    
    % 获取当前图形对象句柄
    fig = gcf;
    
    % 设置图中所有文本的字体为Garamond
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Garamond')
    
    % 设置图中所有文本的字体样式为斜体
    set(findall(fig, '-property', 'FontAngle'), 'FontAngle', 'italic')
end
