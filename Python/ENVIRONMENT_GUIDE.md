# Python 环境与依赖管理指南

Python 新手很多时间不是卡在语法，而是卡在“我到底运行的是哪个 Python”“库到底装到哪里了”。这篇文档专门解释环境和依赖。

## 先确认 Python 版本

在终端运行：

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

第一条看版本，第二条看当前解释器路径。

如果 Windows 上 `python` 不可用，可以试：

```powershell
py --version
```

## 为什么推荐虚拟环境

虚拟环境解决的是“不同项目依赖互相影响”的问题。

例如：

- 项目 A 需要旧版本库。
- 项目 B 需要新版本库。
- 全局安装会互相打架。
- 虚拟环境可以隔离它们。

## 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

激活后，命令行前面通常会出现 `(.venv)`。

如果 PowerShell 不允许执行脚本，可以临时使用：

```powershell
powershell -ExecutionPolicy Bypass
```

再进入项目目录激活环境。

## 安装依赖

推荐写法：

```powershell
python -m pip install requests
```

这样可以保证 `pip` 属于当前 Python。

检查是否安装成功：

```powershell
python -c "import requests; print(requests.__version__)"
```

## 保存依赖列表

如果项目需要多个库，可以生成：

```powershell
python -m pip freeze > requirements.txt
```

别人拿到项目后可以安装：

```powershell
python -m pip install -r requirements.txt
```

新手项目不必一开始追求复杂依赖管理，先把 `requirements.txt` 用明白。

## 常见报错

### `ModuleNotFoundError`

当前 Python 找不到库。先确认解释器路径，再用当前解释器安装：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip install 库名
```

### `pip` 不是内部或外部命令

尝试：

```powershell
python -m pip --version
```

如果可用，以后优先使用 `python -m pip`。

### VS Code 运行和终端运行结果不同

通常是 VS Code 选的解释器不同。

在 VS Code 中打开命令面板，选择 Python 解释器，尽量选项目里的 `.venv`。

## 目录建议

小项目可以这样组织：

```text
project/
├── .venv/
├── main.py
├── requirements.txt
└── README.md
```

`.venv` 不要提交到 GitHub。应该把它写进 `.gitignore`。

## 下一步

环境稳定后，再看 [Python 小项目学习路线](PROJECT_ROADMAP.md)。先把运行方式、依赖和路径打稳，后面写脚本会顺很多。
