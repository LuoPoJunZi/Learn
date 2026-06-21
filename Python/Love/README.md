# Python 表白小程序示例

这个目录保存几个 Python 图形动画小程序，适合练习基础语法、函数拆分、坐标计算和 GUI 绘图。

## 示例列表

| 示例 | 入口文件 | 主要内容 | 使用到的库 |
| :--- | :--- | :--- | :--- |
| LoveYou-03 | [LoveYou-03/main.py](LoveYou-03/main.py) | 使用 `turtle` 绘制爱心和箭头动画 | `turtle`、`time` |
| LoveYou-08 | [LoveYou-08/main.py](LoveYou-08/main.py) | 使用 `tkinter` 绘制动态爱心粒子 | `tkinter`、`random`、`math` |
| LoveYou-09 | [LoveYou-09/main.py](LoveYou-09/main.py) | 使用 `tkinter` 绘制更大画布的爱心效果 | `tkinter`、`random`、`math` |

## 运行环境

建议使用 Python 3.10 或更新版本。

这些示例主要使用 Python 标准库，通常不需要额外安装第三方库。

需要注意：

- `turtle` 会打开图形窗口。
- `tkinter` 也会打开 GUI 窗口。
- 如果你在没有桌面环境的服务器上运行，可能无法显示窗口。

## 如何运行

在 Windows PowerShell 中，可以这样运行：

```powershell
cd Python\Love\LoveYou-03
python main.py
```

运行其他示例时，把目录换成对应名称：

```powershell
cd Python\Love\LoveYou-08
python main.py
```

```powershell
cd Python\Love\LoveYou-09
python main.py
```

如果打开窗口后没有立刻看到完整效果，可以等待几秒，部分动画需要逐步绘制。

## 适合练习什么

### LoveYou-03

适合练习：

- `turtle` 画笔移动
- 函数拆分
- 坐标定位
- 简单动画节奏控制

可以尝试修改：

- 爱心颜色
- 箭头大小
- 绘制速度
- 展示文字

### LoveYou-08 和 LoveYou-09

适合练习：

- `tkinter` 窗口和画布
- 数学函数生成坐标
- 随机数制造粒子效果
- 动画循环刷新

可以尝试修改：

- `CANVAS_WIDTH` 和 `CANVAS_HEIGHT`
- `HEART_COLOR`
- `IMAGE_ENLARGE`
- 粒子数量
- 窗口标题和展示文字

## 常见问题

### 运行后没有窗口

先确认你是在本地电脑运行，而不是没有图形界面的服务器环境。

如果使用的是远程终端、SSH 或某些在线运行环境，`turtle` 和 `tkinter` 可能无法正常显示。

### 中文显示异常

如果你修改了窗口文字或注释后出现乱码，确认文件使用 UTF-8 编码保存。

### 关闭窗口后程序报错

图形动画关闭窗口时偶尔会出现事件循环相关提示。只要窗口能正常显示，一般不影响学习。

## 修改建议

新手建议按这个顺序改：

1. 先改颜色、文字、窗口大小。
2. 再改速度、粒子数量、画笔粗细。
3. 最后再改函数和动画逻辑。

每次只改一小处，运行确认效果，再继续下一步。这样最容易知道是哪一行导致了变化。

