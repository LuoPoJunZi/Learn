# Python 学习与自动化脚本

这个目录整理 Python 基础教程、自动化脚本、算法示例、图片抓取示例和趣味小程序。

## 目录说明

### 学习导读

[Python 学习导读](LEARNING_GUIDE.md) 串起小项目路线、虚拟环境、命令行参数、文件处理、数据处理、爬虫安全边界和脚本打包思路。

[Python 常见问题](FAQ.md) 汇总解释解释器、依赖安装、虚拟环境、相对路径、双击运行、数据结构、爬虫和批量文件处理常见疑问。

### 外部资源导读

[优秀开源仓库导读](EXTERNAL_REPOSITORIES.md) 介绍了几个适合初学者继续学习的 Python 仓库：

- [walter201230/Python](https://github.com/walter201230/Python)：小白 Python 教程
- [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)：算法和数据结构实现
- [geekcomputers/Python](https://github.com/geekcomputers/Python)：实用脚本和自动化示例

### Basics

[Basics](Basics/README.md) 是写给新手的 Python 基础教程，包含：

- 安装、运行与编辑器
- 变量、注释、输入输出
- 常用数据类型
- 条件判断与循环
- 函数与模块
- 文件读写与异常处理
- 虚拟环境与第三方库
- 新手练习路线

如果你还不熟悉 Python 语法，建议先从这里开始。

### Auto_scripts

[Auto_scripts](Auto_scripts/README.md) 是日常自动化脚本集合，包含文件整理、网站检测、图片处理、PDF/Excel 处理、邮件发送等工具。

推荐先看 [Auto_scripts 脚本索引](Auto_scripts/README.md)，按用途选择脚本。

### Neural Network

[Neural Network](<Neural Network/README.md>) 保存 Python 神经网络入门示例。

当前示例：

- 线性回归梯度下降
- 感知机二分类
- 逻辑回归二分类
- 简单 MLP 学习 XOR

这些示例优先使用 Python 标准库，适合先理解原理，再继续学习 NumPy、scikit-learn 或 PyTorch。

### Multi-Objective Optimization

[Multi-Objective Optimization](<Multi-Objective Optimization/README.md>) 保存 Python 多目标优化入门示例。

当前示例：

- 加权和网格搜索
- 简化 Pareto 前沿筛选
- 迷你 NSGA-II
- 简化 MOEA/D

这些示例适合和 Matlab 多目标优化目录对照学习。

### Grab

[Grab](Grab/README.md) 保存图片抓取相关脚本。

当前入口：

- [图片抓取示例说明](Grab/README.md)
- [Grabpicture.py](Grab/Grabpicture.py)

运行前建议先检查目标网站规则、请求频率和保存路径。

### Love

[Love](Love/README.md) 保存 Python 表白小程序示例。

这些示例更偏趣味和学习用途，适合练习图形界面、动画或基础脚本结构。

## 运行环境

建议使用 Python 3.10 或更新版本。

常见依赖可能包括：

- `requests`
- `pandas`
- `openpyxl`
- `Pillow`
- `PyPDF2` 或其他 PDF 处理库
- `tweepy`

算法入门示例目前优先使用标准库。具体依赖以每个脚本目录下的 README 和代码 import 为准。

## 运行脚本前

1. 进入对应脚本目录。
2. 阅读该目录下的 `README.md`。
3. 检查脚本顶部的 import，确认依赖已安装。
4. 检查输入路径、输出路径、账号凭证等配置。
5. 涉及网络请求、邮件、社交平台 API 时，不要把 Token、密码写入仓库。

## 安全提醒

- 爬取网站前先确认目标站点是否允许访问。
- 下载图片或文件时注意版权和来源。
- 邮件脚本不要直接写真实邮箱密码，优先使用环境变量或应用专用密码。
- Twitter/X API、SMTP 密码等敏感信息不要提交到 GitHub。
- 批量删除、移动、重命名文件前，先在测试目录运行。
