function z = Plot_ZDT1()
    % 定义目标函数，使用 ZDT3 函数
    ObjectiveFunction = @(x) ZDT3(x);
    
    % 定义 x 的取值范围
    x = 0:0.01:1;  % 从0到1，以0.01为步长
    
    % 计算目标函数的值
    for i = 1:size(x, 2)
        TPF(i, :) = ObjectiveFunction([x(i) 0 0 0 0]);  % 计算目标函数的目标值
    end
    
    % 绘制目标值的曲线
    line(TPF(:, 1), TPF(:, 2));
    
    % 设置图形的标题和坐标轴标签
    title('ZDT1');
    xlabel('f1');
    ylabel('f2');
    box on;  % 添加边框

    % 获取当前图形并设置字体属性
    fig = gcf;
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Garamond');  % 设置字体为Garamond
    set(findall(fig, '-property', 'FontAngle'), 'FontAngle', 'italic');  % 设置字体为斜体
end
