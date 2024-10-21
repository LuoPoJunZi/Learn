import random  # 导入随机数生成库
from math import sin, cos, pi, log  # 导入数学库中的三角函数和对数
from tkinter import *  # 导入Tkinter库用于创建图形用户界面

# 画布的宽和高
CANVAS_WIDTH = 840  # 画布的宽度
CANVAS_HEIGHT = 680  # 画布的高度
# 画布中心的坐标
CANVAS_CENTER_X = CANVAS_WIDTH / 2  # 画布中心的X轴坐标
CANVAS_CENTER_Y = CANVAS_HEIGHT / 2  # 画布中心的Y轴坐标
IMAGE_ENLARGE = 11  # 放大比例

# 爱心的颜色
HEART_COLOR = "RED"  # 引号内修改颜色！

def heart_function(t, shrink_ratio: float = IMAGE_ENLARGE):
    """
    “爱心函数生成器” - 根据参数t生成爱心形状的坐标
    :param shrink_ratio: 放大比例
    :param t: 参数
    :return: 坐标 (x, y)
    """
    # 基础爱心函数
    x = 17 * (sin(t) ** 3)  # 计算X坐标
    y = -(16 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(3 * t))  # 计算Y坐标

    # 放大坐标
    x *= IMAGE_ENLARGE  # 将X坐标按放大比例缩放
    y *= IMAGE_ENLARGE  # 将Y坐标按放大比例缩放

    # 将坐标移到画布中央
    x += CANVAS_CENTER_X  # X坐标平移至画布中心
    y += CANVAS_CENTER_Y  # Y坐标平移至画布中心

    return int(x), int(y)  # 返回整数坐标

def scatter_inside(x, y, beta=0.15):
    """
    随机内部扩散 - 在爱心内部进行随机扩散
    :param x: 原x坐标
    :param y: 原y坐标
    :param beta: 扩散强度
    :return: 新坐标 (x, y)
    """
    # 计算扩散比例
    ratio_x = - beta * log(random.random())  # 随机生成X方向的扩散量
    ratio_y = - beta * log(random.random())  # 随机生成Y方向的扩散量

    # 计算新的坐标偏移量
    dx = ratio_x * (x - CANVAS_CENTER_X)  # 计算X方向的偏移
    dy = ratio_y * (y - CANVAS_CENTER_Y)  # 计算Y方向的偏移

    return x - dx, y - dy  # 返回新的坐标

def shrink(x, y, ratio):
    """
    抖动 - 通过抖动效果调整坐标
    :param x: 原x坐标
    :param y: 原y坐标
    :param ratio: 抖动比例
    :return: 新坐标 (x, y)
    """
    # 计算力的影响
    force = -1 / (((x - CANVAS_CENTER_X) ** 2 + (y - CANVAS_CENTER_Y) ** 2) ** 0.6)  # 使用魔法参数
    # 计算新的坐标偏移量
    dx = ratio * force * (x - CANVAS_CENTER_X)  # X方向的抖动
    dy = ratio * force * (y - CANVAS_CENTER_Y)  # Y方向的抖动
    
    return x - dx, y - dy  # 返回新的坐标

def curve(p):
    """
    自定义曲线函数 - 调整跳动周期
    :param p: 参数
    :return: 根据正弦函数计算的值
    """
    # 可以尝试换其他的动态函数，达到更有力量的效果（贝塞尔？）
    return 2 * (2 * sin(4 * p)) / (2 * pi)  # 生成动态曲线

class Heart:
    """
    爱心类 - 用于生成和管理爱心的点
    """

    def __init__(self, generate_frame=20):
        self._points = set()  # 原始爱心坐标集合
        self._edge_diffusion_points = set()  # 边缘扩散效果点坐标集合
        self._center_diffusion_points = set()  # 中心扩散效果点坐标集合
        self.all_points = {}  # 每帧动态点坐标
        self.build(2000)  # 构建爱心形状的点
        self.random_halo = 1000  # 随机光环的数量
        self.generate_frame = generate_frame  # 生成帧数
        for frame in range(generate_frame):
            self.calc(frame)  # 计算每一帧的点

    def build(self, number):
        """
        构建爱心的点
        :param number: 生成的点的数量
        """
        # 生成爱心的基本形状
        for _ in range(number):
            t = random.uniform(0, 2 * pi)  # 随机生成t值
            x, y = heart_function(t)  # 根据t生成坐标
            self._points.add((x, y))  # 添加坐标到集合中

        # 爱心边缘扩散
        for _x, _y in list(self._points):  # 遍历所有原始点
            for _ in range(3):  # 每个点扩散3次
                x, y = scatter_inside(_x, _y, 0.05)  # 进行扩散
                self._edge_diffusion_points.add((x, y))  # 添加到边缘扩散集合

        # 爱心中心再次扩散
        point_list = list(self._points)  # 转换为列表以便随机选择
        for _ in range(10000):  # 进行10000次扩散
            x, y = random.choice(point_list)  # 随机选择一个点
            x, y = scatter_inside(x, y, 0.27)  # 进行扩散
            self._center_diffusion_points.add((x, y))  # 添加到中心扩散集合

    @staticmethod
    def calc_position(x, y, ratio):
        """
        计算新坐标 - 调整坐标以产生抖动效果
        :param x: 原x坐标
        :param y: 原y坐标
        :param ratio: 抖动比例
        :return: 新坐标 (x, y)
        """
        # 计算力的影响
        force = 1 / (((x - CANVAS_CENTER_X) ** 2 + (y - CANVAS_CENTER_Y) ** 2) ** 0.420)  # 魔法参数
        # 计算新的坐标偏移量
        dx = ratio * force * (x - CANVAS_CENTER_X) + random.randint(-1, 1)  # X方向抖动
        dy = ratio * force * (y - CANVAS_CENTER_Y) + random.randint(-1, 1)  # Y方向抖动
        
        return x - dx, y - dy  # 返回新的坐标

    def calc(self, generate_frame):
        """
        计算每一帧的动态点
        :param generate_frame: 当前帧数
        """
        ratio = 15 * curve(generate_frame / 10 * pi)  # 计算圆滑的周期的缩放比例
        halo_radius = int(4 + 6 * (1 + curve(generate_frame / 10 * pi)))  # 计算光环半径
        halo_number = int(3000 + 4000 * abs(curve(generate_frame / 10 * pi) ** 2))  # 计算光环数量

        all_points = []  # 存储所有点的列表

        # 生成光环
        heart_halo_point = set()  # 光环的点坐标集合
        for _ in range(halo_number):
            t = random.uniform(0, 2 * pi)  # 随机生成t值
            x, y = heart_function(t, shrink_ratio=-15)  # 计算光环坐标
            x, y = shrink(x, y, halo_radius)  # 进行抖动处理
            if (x, y) not in heart_halo_point:  # 确保坐标不重复
                heart_halo_point.add((x, y))  # 添加到光环坐标集合
                # 处理新的点
                x += random.randint(-60, 60)  # 添加随机偏移
                y += random.randint(-60, 60)  # 添加随机偏移
                size = random.choice((1, 1, 2))  # 随机选择光环的大小
                # 将光环的多个点添加到所有点列表
                all_points.append((x, y, size))
                all_points.append((x + 20, y + 20, size))
                all_points.append((x - 20, y - 20, size))
                all_points.append((x + 20, y - 20, size))
                all_points.append((x - 20, y + 20, size))

        # 计算轮廓点
        for x, y in self._points:  # 遍历原始爱心坐标
            x, y = self.calc_position(x, y, ratio)  # 调整坐标
            size = random.randint(1, 3)  # 随机选择轮廓大小
            all_points.append((x, y, size))  # 添加到所有点列表

        # 计算边缘扩散点
        for x, y in self._edge_diffusion_points:  # 遍历边缘扩散点
            x, y = self.calc_position(x, y, ratio)  # 调整坐标
            size = random.randint(1, 2)  # 随机选择边缘大小
            all_points.append((x, y, size))  # 添加到所有点列表

        # 计算中心扩散点
        for x, y in self._center_diffusion_points:  # 遍历中心扩散点
            x, y = self.calc_position(x, y, ratio)  # 调整坐标
            size = random.randint(1, 2)  # 随机选择中心大小
            all_points.append((x, y, size))  # 添加到所有点列表

        # 将当前帧的点存入字典
        self.all_points[generate_frame] = all_points

    def render(self, render_canvas, render_frame):
        """
        渲染图形 - 将当前帧的所有点绘制到画布上
        :param render_canvas: 目标画布
        :param render_frame: 当前帧数
        """
        # 遍历当前帧的所有点并绘制
        for x, y, size in self.all_points[render_frame % self.generate_frame]:
            render_canvas.create_rectangle(x, y, x + size, y + size, width=0, fill=HEART_COLOR)  # 绘制矩形

def draw(main: Tk, render_canvas: Canvas, render_heart: Heart, render_frame=0):
    """
    绘制函数 - 更新画布内容
    :param main: 主窗口
    :param render_canvas: 渲染画布
    :param render_heart: 爱心对象
    :param render_frame: 当前帧数
    """
    render_canvas.delete('all')  # 清空画布
    render_heart.render(render_canvas, render_frame)  # 渲染当前帧的爱心
    main.after(1, draw, main, render_canvas, render_heart, render_frame + 1)  # 设置下次绘制的时间

if __name__ == '__main__':
    root = Tk()  # 创建主窗口
    root.title("晚上星月争辉，美梦陪你入睡")  # 设置窗口标题
    canvas = Canvas(root, bg='black', height=CANVAS_HEIGHT, width=CANVAS_WIDTH)  # 创建画布
    canvas.pack()  # 将画布放入窗口
    heart = Heart()  # 创建爱心对象
    draw(root, canvas, heart)  # 开始绘制
    root.mainloop()  # 进入Tkinter主循环
