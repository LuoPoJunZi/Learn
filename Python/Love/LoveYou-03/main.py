import turtle  # 导入海龟绘图库
import time  # 导入时间库以便使用延时功能

# 清屏函数
def clear_all():
    turtle.penup()  # 抬起画笔
    turtle.goto(0, 0)  # 移动到坐标原点(0, 0)
    turtle.color('white')  # 设置画笔颜色为白色
    turtle.pensize(800)  # 设置画笔大小为800
    turtle.pendown()  # 放下画笔
    turtle.setheading(0)  # 设置海龟朝向为0度（向右）
    turtle.fd(300)  # 向前移动300单位
    turtle.bk(600)  # 向后移动600单位（清除画面）

# 重定位海龟的位置
def go_to(x, y, state):
    # 根据state决定是否放下画笔
    turtle.pendown() if state else turtle.penup()
    turtle.goto(x, y)  # 移动到指定坐标(x, y)

# 画线
# state为真时海龟回到原点，为假时不回到原来的出发点
def draw_line(length, angle, state):
    turtle.pensize(1)  # 设置画笔大小为1
    turtle.pendown()  # 放下画笔
    turtle.setheading(angle)  # 设置海龟的朝向为指定角度
    turtle.fd(length)  # 向前绘制指定长度的线
    turtle.bk(length) if state else turtle.penup()  # 如果state为真，则向后移动相同长度并抬起画笔
    turtle.penup()  # 抬起画笔以准备下一步操作

# 画箭羽
def draw_feather(size):
    angle = 30  # 箭的倾角
    feather_num = size // 6  # 羽毛的数量
    feather_length = size // 3  # 羽毛的长度
    feather_gap = size // 10  # 羽毛的间隔
    for i in range(feather_num):
        draw_line(feather_gap, angle + 180, False)  # 画箭柄，不折返
        draw_line(feather_length, angle + 145, True)  # 画羽翼，要折返
    draw_line(feather_length, angle + 145, False)  # 画最后一根羽翼，不折返
    draw_line(feather_num * feather_gap, angle, False)  # 画箭杆，不折返
    draw_line(feather_length, angle + 145 + 180, False)  # 画另一边的羽翼
    for i in range(feather_num):
        draw_line(feather_gap, angle + 180, False)  # 画箭柄，不折返
        draw_line(feather_length, angle - 145, True)  # 画羽翼，要折返
    draw_line(feather_length, angle - 145, False)  # 画最后一根羽翼
    draw_line(feather_num * feather_gap, angle, False)  # 画箭杆，不折返
    draw_line(feather_length, angle - 145 + 180, False)  # 画另一边的羽翼

# 画爱心
def draw_heart(size):
    turtle.color('red', 'pink')  # 设置画笔颜色为红色，填充颜色为粉色
    turtle.pensize(2)  # 设置画笔大小为2
    turtle.pendown()  # 放下画笔
    turtle.setheading(150)  # 设置海龟朝向为150度
    turtle.begin_fill()  # 开始填充
    turtle.fd(size)  # 向前绘制指定大小的线
    turtle.circle(size * -3.745, 45)  # 绘制爱心的曲线
    turtle.circle(size * -1.431, 165)  # 绘制爱心的曲线
    turtle.left(120)  # 向左转120度
    turtle.circle(size * -1.431, 165)  # 绘制爱心的曲线
    turtle.circle(size * -3.745, 45)  # 绘制爱心的曲线
    turtle.fd(size)  # 向前绘制指定大小的线
    turtle.end_fill()  # 结束填充

# 画爱心的圆弧
def hart_arc():
    for i in range(200):
        turtle.right(1)  # 向右转1度
        turtle.forward(2)  # 向前移动2单位

# 画箭
def draw_arrow(size):
    angle = 30  # 箭的倾角
    turtle.color('black')  # 设置画笔颜色为黑色
    draw_feather(size)  # 画箭羽
    turtle.pensize(4)  # 设置画笔大小为4
    turtle.setheading(angle)  # 设置海龟朝向为30度
    turtle.pendown()  # 放下画笔
    turtle.fd(size * 2)  # 向前绘制箭身

# 一箭穿心
# 箭的头没有画出来，而是用海龟来代替
def arrow_heart(x, y, size):
    go_to(x, y, False)  # 移动到指定位置
    draw_heart(size * 1.15)  # 画较大的爱心
    turtle.setheading(-150)  # 设置海龟朝向为-150度
    turtle.penup()  # 抬起画笔
    turtle.fd(size * 2.2)  # 向前移动
    draw_heart(size)  # 画较小的爱心
    turtle.penup()  # 抬起画笔
    turtle.setheading(150)  # 设置海龟朝向为150度
    turtle.fd(size * 2.2)  # 向前移动
    draw_arrow(size)  # 画箭

# 画出发射爱心的小人
def draw_people(x, y):
    turtle.penup()  # 抬起画笔
    turtle.goto(x, y)  # 移动到指定位置
    turtle.pendown()  # 放下画笔
    turtle.pensize(2)  # 设置画笔大小为2
    turtle.color('black')  # 设置画笔颜色为黑色
    turtle.setheading(0)  # 设置海龟朝向为0度
    turtle.circle(60, 360)  # 画一个圆形，代表头部
    turtle.penup()  # 抬起画笔
    turtle.setheading(90)  # 设置朝向为90度
    turtle.fd(75)  # 向上移动75单位
    turtle.setheading(180)  # 设置朝向为180度
    turtle.fd(20)  # 向左移动20单位
    turtle.pensize(4)  # 设置画笔大小为4
    turtle.pendown()  # 放下画笔
    turtle.circle(2, 360)  # 画一个小圆形，代表眼睛
    turtle.setheading(0)  # 设置朝向为0度
    turtle.penup()  # 抬起画笔
    turtle.fd(40)  # 向前移动40单位
    turtle.pensize(4)  # 设置画笔大小为4
    turtle.pendown()  # 放下画笔
    turtle.circle(-2, 360)  # 画另一个小圆形，代表另一只眼睛
    turtle.penup()  # 抬起画笔
    turtle.goto(x, y)  # 回到初始位置
    turtle.setheading(-90)  # 设置朝向为-90度
    turtle.pendown()  # 放下画笔
    turtle.fd(20)  # 向下移动20单位
    turtle.setheading(0)  # 设置朝向为0度
    turtle.fd(35)  # 向前移动35单位
    turtle.setheading(60)  # 设置朝向为60度
    turtle.fd(10)  # 向前移动10单位
    turtle.penup()  # 抬起画笔
    turtle.goto(x, y)  # 回到初始位置
    turtle.setheading(-90)  # 设置朝向为-90度
    turtle.pendown()  # 放下画笔
    turtle.fd(40)  # 向下移动40单位
    turtle.setheading(0)  # 设置朝向为0度
    turtle.fd(35)  # 向前移动35单位
    turtle.setheading(-60)  # 设置朝向为-60度
    turtle.fd(10)  # 向前移动10单位
    turtle.penup()  # 抬起画笔
    turtle.goto(x, y)  # 回到初始位置
    turtle.setheading(-90)  # 设置朝向为-90度
    turtle.pendown()  # 放下画笔
    turtle.fd(60)  # 向下移动60单位
    turtle.setheading(-135)  # 设置朝向为-135度
    turtle.fd(60)  # 向前移动60单位
    turtle.bk(60)  # 向后移动60单位
    turtle.setheading(-45)  # 设置朝向为-45度
    turtle.fd(30)  # 向前移动30单位
    turtle.setheading(-135)  # 设置朝向为-135度
    turtle.fd(35)  # 向前移动35单位
    turtle.penup()  # 抬起画笔

# 第一个画面，显示文字
def page0():
    turtle.penup()  # 抬起画笔
    turtle.goto(-350, 0)  # 移动到指定位置
    turtle.color('black')  # 设置画笔颜色为黑色
    turtle.write('专属于我们的情人节', font=('宋体', 60, 'normal'))  # 写出文字
    time.sleep(3)  # 暂停3秒

# 第二个画面，显示发射爱心的小人
def page1():
    turtle.speed(10)  # 设置海龟速度为10
    draw_people(-250, 20)  # 画小人
    turtle.penup()  # 抬起画笔
    turtle.goto(-150, -30)  # 移动到指定位置
    draw_heart(14)  # 画小爱心
    turtle.penup()  # 抬起画笔
    turtle.goto(-20, -60)  # 移动到指定位置
    draw_heart(25)  # 画中等大小的爱心
    turtle.penup()  # 抬起画笔
    turtle.goto(250, -100)  # 移动到指定位置
    draw_heart(45)  # 画较大的爱心
    turtle.hideturtle()  # 隐藏海龟
    time.sleep(3)  # 暂停3秒

# 最后一个画面，一箭穿心
def page2():
    turtle.speed(1)  # 设置海龟速度为1
    turtle.penup()  # 抬起画笔
    turtle.goto(-200, -200)  # 移动到指定位置
    turtle.color('blue')  # 设置画笔颜色为蓝色
    turtle.pendown()  # 放下画笔
    turtle.write('大笨蛋         小笨蛋', font=('wisdom', 50, 'normal'))  # 写出文字
    turtle.penup()  # 抬起画笔
    turtle.goto(0, -190)  # 移动到指定位置
    draw_heart(10)  # 画小爱心
    arrow_heart(20, -60, 51)  # 画箭穿心效果
    turtle.showturtle()  # 显示海龟

# 主函数
def main():
    turtle.setup(900, 500)  # 设置画布大小为900x500
    page0()  # 显示第一个画面
    clear_all()  # 清屏
    page1()  # 显示第二个画面
    clear_all()  # 清屏
    page2()  # 显示最后一个画面
    turtle.done()  # 完成绘图，保持窗口开启

main()  # 调用主函数
