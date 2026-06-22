# Python 优秀开源仓库导读

这份文档给初学者介绍几个值得收藏的 Python 开源仓库。它们不是本仓库内容的替代品，而是适合作为“课外阅读”和“练习素材”的资源。

学习建议：

1. 先学本仓库的 [Python 基础教程](Basics/README.md)。
2. 再看这些外部仓库，选择适合自己阶段的内容。
3. 不要一口气收藏很多资料却不动手练习。
4. 看到好代码时，先运行，再改一小处，最后自己重写一遍。

## 推荐仓库总览

| 仓库 | 适合阶段 | 主要用途 | 建议看法 |
| :--- | :--- | :--- | :--- |
| [walter201230/Python](https://github.com/walter201230/Python) | 零基础到入门 | 系统学习 Python 基础语法 | 按章节通读，配合练习 |
| [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) | 入门后到进阶 | 学习算法和数据结构实现 | 按主题查算法，逐个复现 |
| [geekcomputers/Python](https://github.com/geekcomputers/Python) | 入门后练手 | 学习实用脚本和自动化小工具 | 挑一个脚本运行并改造 |

## walter201230/Python

仓库地址：

- [https://github.com/walter201230/Python](https://github.com/walter201230/Python)

这个仓库定位是“小白 Python 教程”。它的 README 中说明教程基于 Python 3.10+，并提供互动版和文档版学习入口。

适合：

- 完全零基础的新手。
- 想系统补 Python 基础语法的人。
- 学完一章就想马上做练习的人。

建议学习方式：

1. 按章节顺序看，不要跳太快。
2. 每学一个语法点，就自己写一个小例子。
3. 对照本仓库 [Python/Basics](Basics/README.md)，把不熟的知识点补一遍。
4. 用 [学习记录模板](../docs/STUDY_NOTES_TEMPLATE.md) 记录每天学了什么。

重点关注：

- Python 安装和第一个程序
- 基本数据类型
- 条件语句和循环
- 函数
- 面向对象
- 模块与包
- 异常处理
- 工程化基础

## TheAlgorithms/Python

仓库地址：

- [https://github.com/TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)

这个仓库定位是“用 Python 实现各种算法”。它的 README 明确说明这些实现主要用于学习，可能不如 Python 标准库中的实现高效。

适合：

- 已经学完 Python 基础语法的人。
- 想学习算法和数据结构的人。
- 准备刷题、竞赛、面试或补计算机基础的人。

建议学习方式：

1. 不要从头到尾硬啃。
2. 先选一个主题，例如排序、搜索、图算法、数学算法。
3. 先读 README 或目录索引，再打开具体算法文件。
4. 自己用小数据跑一遍。
5. 尝试不看源码重写一次。

适合优先看的主题：

- 排序算法
- 搜索算法
- 动态规划
- 图算法
- 数学算法
- 字符串算法
- 机器学习和神经网络相关目录

和本仓库的关系：

- 本仓库更偏“教程 + 小白解释 + 入门示例”。
- TheAlgorithms/Python 更像“算法代码大全”。
- 学完本仓库 [Python 多目标优化示例](<Multi-Objective Optimization/README.md>) 和 [Python 神经网络示例](<Neural Network/README.md>) 后，可以去 TheAlgorithms/Python 查更多算法实现。

## geekcomputers/Python

仓库地址：

- [https://github.com/geekcomputers/Python](https://github.com/geekcomputers/Python)

这个仓库收集了很多 Python 小脚本。README 中说明这些脚本用于减少人工工作量，也适合作为初学者学习 Python 的教育示例。

适合：

- 学完基础语法后想找脚本练手的人。
- 想学习文件处理、图片处理、下载工具、小游戏、自动化任务的人。
- 想从“语法学习”过渡到“解决实际小问题”的人。

建议学习方式：

1. 先挑一个你能看懂用途的脚本。
2. 阅读脚本顶部的 import，判断依赖。
3. 在测试目录里运行，不要直接对重要文件夹操作。
4. 修改一个小参数，比如路径、文件名、输出格式。
5. 最后尝试把脚本改成自己的小工具。

适合优先看的方向：

- 批量重命名
- 自动创建目录
- 图片下载
- 文件整理
- 小游戏或图形示例

和本仓库的关系：

- 本仓库 [Auto_scripts](Auto_scripts/README.md) 已经整理了一批日常自动化脚本。
- geekcomputers/Python 可以作为更多脚本灵感来源。
- 看到涉及下载、批量删除、批量移动、账号登录的脚本时，先读代码再运行。

## 初学者使用外部仓库的注意事项

### 不要直接运行不了解的脚本

尤其是这些类型：

- 删除文件
- 移动文件
- 批量重命名
- 下载大量内容
- 登录账号
- 发送邮件
- 调用 API

先读 README，再读代码，最后在测试目录运行。

### 不要只收藏不练习

收藏 100 个仓库，不如认真跑通 3 个脚本。

推荐节奏：

```text
看说明 -> 运行 -> 改参数 -> 加注释 -> 自己重写
```

### 注意许可证

不同仓库许可证不同。学习、引用、改写、转载前，先看对方仓库的 LICENSE。

如果只是自己学习，一般问题不大；如果要复制到自己的项目、博客或商业项目中，需要遵守对方许可证要求。

## 推荐学习路线

1. 本仓库 [Python 基础教程](Basics/README.md)
2. [walter201230/Python](https://github.com/walter201230/Python)：补系统语法
3. 本仓库 [Auto_scripts](Auto_scripts/README.md)：跑自动化脚本
4. [geekcomputers/Python](https://github.com/geekcomputers/Python)：找更多脚本灵感
5. 本仓库 [Python 神经网络示例](<Neural Network/README.md>) 和 [Python 多目标优化示例](<Multi-Objective Optimization/README.md>)
6. [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)：系统补算法

## 参考来源

- [walter201230/Python](https://github.com/walter201230/Python)
- [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)
- [geekcomputers/Python](https://github.com/geekcomputers/Python)

