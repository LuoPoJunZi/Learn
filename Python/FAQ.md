# Python 常见问题

这份 FAQ 面向刚开始写 Python 脚本的新手，重点解释运行环境、依赖、路径和报错。

## 为什么提示 `python` 不是内部或外部命令？

常见原因：

- 没有安装 Python。
- 安装时没有加入 PATH。
- Windows 上需要使用 `py` 命令。

可以尝试：

```powershell
python --version
py --version
```

如果两个都不可用，先重新检查 Python 安装。

## 为什么提示 `ModuleNotFoundError`？

这表示当前 Python 环境找不到某个库。

先确认当前解释器：

```powershell
python -c "import sys; print(sys.executable)"
```

再安装依赖：

```powershell
python -m pip install requests
```

推荐使用 `python -m pip`，这样安装位置更容易和当前解释器对应。

## 虚拟环境是不是必须的？

不是必须，但强烈推荐。虚拟环境可以让每个项目有自己的依赖，避免不同项目互相影响。

常见创建方式：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

激活后再安装依赖。

## 为什么相对路径找不到文件？

相对路径是相对于“当前工作目录”，不一定是脚本所在目录。

运行前先看：

```python
from pathlib import Path
print(Path.cwd())
```

写脚本时，建议用 `pathlib` 处理路径，少拼字符串。

## 为什么双击 `.py` 文件一闪而过？

脚本运行完窗口就关闭了。新手阶段建议在终端中运行：

```powershell
python script.py
```

这样能看到输出和报错。

## 什么时候用列表、字典、集合？

简单理解：

- 列表 `list`：有顺序、可重复，例如一组文件名。
- 字典 `dict`：键值对，例如姓名到分数。
- 集合 `set`：去重，例如唯一的标签集合。

数据结构选对了，代码会简单很多。

## 爬虫脚本可以随便跑吗？

不建议。跑网络请求脚本前要确认：

- 网站是否允许抓取。
- 是否需要登录或权限。
- 请求频率是否过高。
- 保存的数据是否涉及版权或隐私。

不确定时，先降低频率，只抓少量公开页面测试。

## 批量处理文件前要注意什么？

先在测试目录运行。尤其是删除、重命名、移动、覆盖文件时，建议先打印计划，不要直接执行。

示例思路：

```python
for path in files:
    print("将要处理:", path)
```

确认输出无误后，再加入真正的操作。

## 下一步

如果你总是卡在环境和运行方式上，先看 [Python 学习导读](LEARNING_GUIDE.md) 和 [虚拟环境与第三方库](Basics/1.7-venv-packages.md)。如果想找实用脚本，从 [Auto_scripts](Auto_scripts/README.md) 开始。
