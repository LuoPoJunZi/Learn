# Learn 新手学习路线

这份路线是给第一次打开这个仓库的新手看的。你不需要一次学完所有目录，先按顺序建立“会写文档、会上传、会运行、会改代码”的基本能力，再回头深入专题。

## 适合谁

- 不熟悉 GitHub，不知道仓库、提交、推送是什么意思。
- 会一点电脑操作，但不太会命令行。
- 想学习 Markdown、Linux、Python、HTML、Matlab、Mathematica 等工具。
- 想把这个仓库当成长期知识库，而不是一次性资料包。

## 先记住一条主线

学习顺序建议是：

```text
Markdown 写文档
-> GitHub 管理文档和代码
-> Linux 理解命令行和服务器
-> Python 写自动化脚本
-> HTML 改网页示例
-> Matlab / Mathematica 做计算和建模
```

这个顺序的好处是：你先学会记录，再学会保存和同步，然后再学会运行环境和编程工具。

## 第一阶段：会读、会写、会改文档

目标：能看懂仓库目录，能自己写一篇清晰的 Markdown 笔记。

建议阅读：

1. [知识地图总览](KNOWLEDGE_MAP.md)
2. [30 天学习计划](30_DAY_PLAN.md)
3. [内容阅读指南](READING_GUIDE.md)
4. [后续深入路线](NEXT_STEP_ROUTES.md)
5. [新手检查清单合集](CHECKLISTS.md)
6. [Markdown 学习导读](../Markdown/LEARNING_GUIDE.md)
7. [Markdown 写作进阶指南](../Markdown/WRITING_GUIDE.md)
8. [README 审查清单](../Markdown/README_REVIEW_CHECKLIST.md)
9. [Markdown 用法教程](../Markdown/README.md)
10. [README 模板](../Markdown/examples/readme-template.md)
11. [文档整理规范](STYLE_GUIDE.md)
12. [新手常见问题排查手册](TROUBLESHOOTING.md)
13. [学习记录模板](STUDY_NOTES_TEMPLATE.md)

你应该练会：

- 使用标题、列表、代码块、表格、链接和图片。
- 给一段命令或代码加上正确的代码块语言。
- 写清楚“这是什么、怎么用、注意什么”。

练习任务：

1. 新建一篇自己的学习笔记。
2. 用 README 模板写一个小项目说明。
3. 检查所有链接是否能点开。
4. 用 [学习记录模板](STUDY_NOTES_TEMPLATE.md) 记录这次练习过程。

更多练习可以看 [新手练习题与自查答案](PRACTICE_EXERCISES.md)。

## 第二阶段：会用 GitHub 保存学习成果

目标：知道 Git 和 GitHub 的区别，能把本地文件提交到远程仓库。

建议阅读：

1. [GitHub 学习导读](../Github/LEARNING_GUIDE.md)
2. [GitHub 真实场景指南](../Github/SCENARIOS.md)
3. [Pull Request Review 入门指南](../Github/REVIEW_GUIDE.md)
4. [Git 与 GitHub 的区别](../Github/docs/part1-basics/1.1-what-is-git.md)
5. [账号注册与安装 Git](../Github/docs/part1-basics/1.2-setup.md)
6. [工作区、暂存区、本地仓库](../Github/docs/part2-git-cmds/2.1-core-concepts.md)
7. [日常 Git 命令](../Github/docs/part2-git-cmds/2.2-daily-commands.md)
8. [第一次提交练习](../Github/docs/part2-git-cmds/2.5-first-commit-practice.md)

你应该练会：

- `git status`
- `git add`
- `git commit`
- `git push`
- 用 VS Code 查看变更并提交。

练习任务：

1. 修改一篇 Markdown 文档。
2. 查看变更。
3. 提交并推送。
4. 在 GitHub 网页上确认文件已经更新。

更多练习可以看 [新手练习题与自查答案](PRACTICE_EXERCISES.md)。

## 第三阶段：会用 Linux 和命令行

目标：能在服务器或终端里完成基础操作，不害怕黑色窗口。

建议阅读：

1. [Linux 学习导读](../Linux/LEARNING_GUIDE.md)
2. [Linux VPS 日常运维导读](../Linux/OPERATIONS_GUIDE.md)
3. [Linux VPS 基础安全清单](../Linux/SECURITY_BASELINE.md)
4. [Linux 新手地图](../Linux/docs/part0-newbie-map.md)
5. [Linux 是什么](../Linux/docs/part1-basics/1.1-linux-intro.md)
6. [文件和目录操作](../Linux/docs/part1-basics/1.2-files-and-directories.md)
7. [文件查看、编辑、搜索](../Linux/docs/part1-basics/1.3-file-view-edit.md)
8. [命令速查附录](../Linux/docs/appendix/command-reference.md)

你应该练会：

- `pwd`、`ls`、`cd`
- `cat`、`less`、`head`、`tail`
- `cp`、`mv`、`mkdir`
- `grep` 或其他搜索工具
- 读懂命令报错的关键字。

练习任务：

1. 创建一个测试目录。
2. 新建一个文本文件。
3. 查看文件内容。
4. 复制、重命名、删除测试文件。

删除或覆盖文件前，请先确认路径，尤其是在服务器环境中。

更多练习可以看 [新手练习题与自查答案](PRACTICE_EXERCISES.md)。

## 第四阶段：用 Python 做自动化

目标：能读懂基础 Python 脚本，并能改简单参数。

建议阅读：

1. [Python 学习导读](../Python/LEARNING_GUIDE.md)
2. [Python 小项目学习路线](../Python/PROJECT_ROADMAP.md)
3. [Python 环境与依赖管理指南](../Python/ENVIRONMENT_GUIDE.md)
4. [Python 基础教程](../Python/Basics/README.md)
5. [安装、运行与编辑器](../Python/Basics/1.1-setup-and-run.md)
6. [变量、输入输出](../Python/Basics/1.2-variables-io.md)
7. [文件读写与异常处理](../Python/Basics/1.6-files-exceptions.md)
8. [虚拟环境与第三方库](../Python/Basics/1.7-venv-packages.md)
9. [Auto_scripts 脚本索引](../Python/Auto_scripts/README.md)

你应该练会：

- 运行一个 `.py` 文件。
- 看懂 `import`、变量、函数和 `if`。
- 安装第三方库。
- 修改输入路径和输出路径。

练习任务：

1. 写一个打印当前时间的小脚本。
2. 写一个统计文本行数的小脚本。
3. 选择一个 `Auto_scripts` 里的脚本，在测试目录中运行。

批量改文件、删文件、发邮件、请求网站前，先用测试数据试一遍。

## 第五阶段：能改静态网页

目标：能看懂 HTML、CSS、JavaScript 各负责什么，并能修改网页文字、图片和样式。

建议阅读：

1. [HTML 基础教程](../HTML/Basics/README.md)
2. [网页结构](../HTML/Basics/1.1-webpage-structure.md)
3. [常用标签](../HTML/Basics/1.2-common-tags.md)
4. [CSS 基础](../HTML/Basics/1.5-css-basics.md)
5. [JavaScript 基础](../HTML/Basics/1.7-javascript-basics.md)

你应该练会：

- 找到页面标题和正文。
- 替换图片地址。
- 修改颜色、字号、间距。
- 看懂简单点击事件。

练习任务：

1. 打开一个 HTML 示例页面。
2. 修改页面标题和一段文字。
3. 修改一个按钮或图片。
4. 在浏览器中刷新查看效果。

## 第六阶段：按专业方向深入

如果你偏数学计算、工程计算、优化算法，可以看：

- [Matlab 学习与算法资料库](../Matlab/README.md)
- [Matlab 学习导读](../Matlab/LEARNING_GUIDE.md)
- [Matlab 案例阅读指南](../Matlab/CASE_READING_GUIDE.md)
- [Matlab 数据与绘图入门指南](../Matlab/DATA_PLOT_GUIDE.md)
- [Matlab 基础教程](../Matlab/Basics/README.md)
- [多目标优化算法索引](<../Matlab/Multi-Objective Optimization/README.md>)
- [神经网络资料索引](<../Matlab/Neural Network/README.md>)

如果你偏符号计算、公式推导、数学可视化，可以看：

- [Mathematica 学习笔记](../Mathematica/README.md)
- [Mathematica 学习导读](../Mathematica/LEARNING_GUIDE.md)
- [Mathematica 专题学习指南](../Mathematica/TOPIC_GUIDE.md)
- [Mathematica Notebook 写作指南](../Mathematica/NOTEBOOK_GUIDE.md)
- [Mathematica 基础教程](../Mathematica/Basics/README.md)

学习建议：

- 先跑通基础示例，再看算法。
- 先理解输入、输出和参数，再尝试修改模型。
- 保留原始示例，复制一份做实验。

## 七天入门安排

如果你想快速开始，可以按下面节奏：

| 天数 | 学习内容 | 产出 |
| --- | --- | --- |
| 第 1 天 | Markdown 基础 | 写一篇学习笔记 |
| 第 2 天 | GitHub 基础 | 完成一次提交和推送 |
| 第 3 天 | Linux 文件命令 | 能在终端中创建、查看、移动文件 |
| 第 4 天 | Python 基础语法 | 写一个可运行的小脚本 |
| 第 5 天 | Python 自动化脚本 | 跑通一个现有脚本 |
| 第 6 天 | HTML/CSS 基础 | 修改一个网页示例 |
| 第 7 天 | Matlab 或 Mathematica | 跑通一个计算示例 |

## 遇到问题时怎么查

先按这个顺序排查：

1. 看当前目录的 `README.md`。
2. 看对应教程里的“常见问题”或“注意事项”。
3. 复制完整报错，重点看第一行和最后几行。
4. 检查路径、文件名、大小写、依赖是否安装。
5. 如果涉及 Git，先运行 `git status`。
6. 如果还是不清楚，查看 [新手常见问题排查手册](TROUBLESHOOTING.md)。

不要一上来就重装环境。很多新手问题只是路径不对、命令位置不对、文件没有保存，或者复制命令时少了一个符号。
