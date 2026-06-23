# Python 学习导读

Python 入门不要只盯着语法。更好的路线是：先能运行脚本，再理解数据结构、函数、文件、依赖，最后把脚本变成能反复使用的小工具。

## 适合人群

- 第一次系统学 Python 的新手。
- 想用 Python 做文件处理、数据处理、爬虫或自动化的人。
- 会复制脚本，但不太会安装依赖和排错的人。

## 学习目标

学完后，你应该能：

- 创建并运行 `.py` 文件。
- 使用虚拟环境管理项目依赖。
- 读懂函数、模块、文件读写和异常处理。
- 写一个带命令行参数的小脚本。
- 知道爬虫和自动化脚本的安全边界。

## 推荐阅读顺序

1. [Python 基础教程](Basics/README.md)
2. [安装、运行与编辑器](Basics/1.1-setup-and-run.md)
3. [变量、输入输出](Basics/1.2-variables-io.md)
4. [函数与模块](Basics/1.5-functions-modules.md)
5. [文件读写与异常处理](Basics/1.6-files-exceptions.md)
6. [虚拟环境与第三方库](Basics/1.7-venv-packages.md)
7. [Auto_scripts 脚本索引](Auto_scripts/README.md)
8. [优秀开源仓库导读](EXTERNAL_REPOSITORIES.md)

## 小项目学习路线

```text
命令行小工具
-> 文件整理脚本
-> 文本统计脚本
-> Excel / CSV 数据处理
-> 图片或 PDF 批处理
-> 网站状态检测
-> 简单机器学习或优化示例
```

每个小项目都先回答四个问题：

- 输入是什么？
- 输出是什么？
- 依赖哪些库？
- 出错时要提示什么？

## 虚拟环境与依赖管理

建议每个项目都有自己的虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install requests
```

常见误区：

- 不是所有电脑都只有一个 Python。
- `pip install` 装到哪里，取决于当前激活的 Python 环境。
- 报 `ModuleNotFoundError` 时，先检查当前解释器和安装位置。

## 命令行参数入门

当脚本不想每次改源码时，可以用命令行参数：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

print(f"处理目录: {args.path}")
```

这样脚本可以运行：

```powershell
python tool.py .\data
```

## 文件处理与数据处理

文件处理脚本要特别注意：

- 先在测试目录运行。
- 批量删除、覆盖、重命名前先打印计划。
- 路径使用 `pathlib` 更稳。
- 处理 CSV、Excel 前先确认编码和表头。

如果任务涉及表格数据，可以从 `pandas` 学起；如果只是读写 Excel，可以先看 `openpyxl` 或现有脚本说明。

## 爬虫安全边界

写网络请求脚本时，要遵守：

- 阅读目标网站规则和 robots 说明。
- 控制请求频率。
- 不抓取隐私、付费或需要绕过权限的内容。
- 保存数据前确认版权和用途。
- 不把 Token、Cookie、密码写入仓库。

## 脚本打包思路

新手阶段不用急着打包成 `.exe`。先做到：

1. 能在命令行运行。
2. 参数可以从命令行传入。
3. 依赖写清楚。
4. README 写明输入输出。

当脚本稳定后，再考虑 `pyinstaller` 这类打包工具。

## 少量自查

- 你知道当前终端使用的是哪个 Python 吗？
- 你能解释虚拟环境解决了什么问题吗？
- 你的脚本出错时，用户能看懂提示吗？
- 批量处理文件前，你是否先用测试目录验证？

## 外部资源

- [Python Tutorial](https://docs.python.org/3/tutorial/)：Python 官方教程。
- [venv 文档](https://docs.python.org/3/library/venv.html)：虚拟环境官方说明。
- [argparse 文档](https://docs.python.org/3/library/argparse.html)：命令行参数官方说明。
- [Python Packaging User Guide](https://packaging.python.org/)：Python 打包和发布指南。
- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/)：数据处理常用库文档。

## 下一步

先从 [Auto_scripts](Auto_scripts/README.md) 选一个风险低的脚本，例如文本统计或网站检测，读懂后改成自己的小工具。别急，能把一个脚本讲明白，比收藏十个脚本更有用。
