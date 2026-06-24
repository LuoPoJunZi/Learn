# Python 小项目学习路线

这篇文档给 Python 新手一条“小项目学习路线”。它不追求项目数量，而是帮助你按能力逐步升级：能运行、能读懂、能改参数、能处理文件、能整理成工具。

## 路线总览

```text
输入输出脚本
-> 文本处理
-> 文件整理
-> CSV / Excel 处理
-> 图片或 PDF 批处理
-> 网站状态检测
-> 命令行小工具
-> 简单算法或模型示例
```

## 阶段一：输入输出脚本

目标：熟悉 `.py` 文件如何运行。

你需要理解：

- `print`
- `input`
- 变量
- 字符串格式化

可以做的小项目：

- 生成一段自我介绍。
- 根据输入的名字输出问候语。
- 计算两个数字的和。

重点不是项目本身，而是掌握“保存文件、打开终端、运行脚本、查看报错”这一套动作。

## 阶段二：文本处理

目标：学会处理字符串和文本文件。

你需要理解：

- 字符串方法。
- `open` 或 `pathlib`。
- `for` 循环。
- 异常处理。

可以参考 [count_words](Auto_scripts/count_words/README.md) 这类脚本。

学习时重点看：

- 输入文件在哪里。
- 输出结果是什么。
- 文件不存在时怎么提示。

## 阶段三：文件整理

目标：学会批量处理文件，但避免误操作。

你需要理解：

- `pathlib.Path`
- 遍历目录。
- 文件扩展名。
- 复制、移动、重命名。

可以参考：

- [sort_files](Auto_scripts/sort_files/README.md)
- [rename_files](Auto_scripts/rename_files/README.md)
- [remove_empty_folders](Auto_scripts/remove_empty_folders/README.md)

安全原则：

- 先打印计划。
- 用测试目录。
- 不要一开始就删除真实文件。

## 阶段四：表格数据处理

目标：理解 CSV / Excel 的读取、筛选和保存。

你需要理解：

- 表头。
- 行和列。
- 编码。
- 缺失值。

可以参考 [read_excel](Auto_scripts/read_excel/README.md)。

如果使用 `pandas`，先学会：

```python
import pandas as pd

df = pd.read_excel("data.xlsx")
print(df.head())
```

不要一开始就写复杂分析。先确认数据读进来是否正确。

## 阶段五：图片和 PDF 批处理

目标：学会调用第三方库处理文件。

可以参考：

- [resize_image](Auto_scripts/resize_image/README.md)
- [extract_text_from_pdf](Auto_scripts/extract_text_from_pdf/README.md)
- [recognize_text](Auto_scripts/recognize_text/README.md)

重点理解：

- 依赖库怎么安装。
- 输入输出路径怎么写。
- 批处理失败时是否跳过单个文件继续执行。

## 阶段六：网络请求和网站检测

目标：理解 HTTP 请求、状态码和请求频率。

可以参考 [check_website_status](Auto_scripts/check_website_status/README.md)。

你需要理解：

- `200`、`404`、`500` 这类状态码。
- 超时。
- 请求头。
- 频率控制。

涉及爬虫时，务必看 [Python 学习导读](LEARNING_GUIDE.md) 里的安全边界说明。

## 阶段七：命令行小工具

目标：让脚本不用改源码也能传参数。

核心工具是 `argparse`：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
```

当脚本可以这样运行时，它就更像一个工具：

```powershell
python tool.py .\data
```

## 阶段八：算法和模型示例

目标：从“工具脚本”过渡到“算法理解”。

可以看：

- [Python 神经网络示例](<Neural Network/README.md>)
- [Python 多目标优化示例](<Multi-Objective Optimization/README.md>)

学习建议：

- 先看输入输出。
- 再看核心公式或循环。
- 最后再改参数。

## 下一步

如果你不知道从哪个项目开始，先选一个不会破坏文件的脚本，例如文本统计或网站检测。跑通一个、改懂一个、写清一个，比同时打开十个项目更有用。
