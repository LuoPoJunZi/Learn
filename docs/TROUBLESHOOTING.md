# 新手常见问题排查手册

这份手册不是替代教程，而是给你在“运行失败、命令报错、页面不显示、脚本没反应”时快速定位问题用的。

排查问题时，先记住一个顺序：

```text
看报错 -> 看路径 -> 看文件名 -> 看依赖 -> 看权限 -> 看网络 -> 再搜索
```

很多新手问题都不是知识点太难，而是路径错了、文件没保存、命令跑错目录、依赖没装、大小写不一致。

## 通用排查方法

### 1. 先复制完整报错

不要只看“失败了”。至少看：

- 第一行报错
- 最后一行报错
- 报错里出现的文件名
- 报错里出现的命令或函数名

如果要问别人，尽量提供：

- 你运行的命令
- 当前目录
- 完整报错
- 你想达到的目标

### 2. 确认当前目录

很多命令依赖“你现在站在哪里”。

Windows PowerShell：

```powershell
pwd
Get-ChildItem
```

Linux / macOS：

```bash
pwd
ls
```

如果文件不在当前目录，命令就可能找不到它。

### 3. 检查文件是否保存

编辑器里改了文件，但没有保存，也会导致运行结果没有变化。

常见表现：

- 页面刷新后还是旧内容。
- Python 运行结果没变化。
- Git 看不到修改。

可以运行：

```bash
git status
```

如果 Git 没显示改动，先确认文件是否保存，或者你是否改错了目录。

### 4. 检查大小写和空格

有些系统区分大小写，有些不区分。路径里有空格时也容易出错。

路径包含空格时，命令里要加引号：

```powershell
cd "Matlab\Neural Network"
```

Markdown 链接里路径包含空格时，建议用尖括号：

```markdown
[Neural Network](<../Matlab/Neural Network/README.md>)
```

## Git / GitHub 常见问题

对应教程：

- [GitHub 新手入门教程](../Github/README.md)
- [日常 Git 命令](../Github/docs/part2-git-cmds/2.2-daily-commands.md)

### git status 显示很多红色文件

含义：这些文件已经修改或新增，但还没有暂存。

可以先查看：

```bash
git status
git diff
```

确认无误后再暂存：

```bash
git add 文件名
```

不要一看到红色就害怕。红色只是“Git 发现了变化，还没准备提交”。

### fatal: not a git repository

含义：你当前目录不是 Git 仓库，或者没有站在仓库里面。

先运行：

```bash
pwd
ls
```

确认当前目录里有没有 `.git`，或者回到仓库根目录。

### nothing to commit, working tree clean

含义：当前没有可提交的改动。

可能原因：

- 文件没有保存。
- 你改的是另一个目录里的文件。
- 改动已经提交过。
- 文件被 `.gitignore` 忽略了。

### push 失败

常见原因：

- 没有登录 GitHub。
- 远程地址不对。
- 当前分支不是 `main`。
- 远程仓库有新提交，本地需要先同步。

先看：

```bash
git remote -v
git branch
git status
```

不要在没看懂提示时强行覆盖远程仓库。

## Python 常见问题

对应教程：

- [Python 基础教程](../Python/Basics/README.md)
- [Python 自动化脚本依赖总览](../Python/Auto_scripts/DEPENDENCIES.md)

### python 不是内部或外部命令

含义：系统找不到 Python 命令。

排查：

- Python 是否安装。
- 安装时是否勾选 Add Python to PATH。
- Windows 可以尝试：

```powershell
py --version
python --version
```

如果 `py` 可以用，运行脚本时也可以写：

```powershell
py script.py
```

### ModuleNotFoundError

含义：缺少第三方库。

例如：

```text
ModuleNotFoundError: No module named 'requests'
```

安装：

```bash
pip install requests
```

如果你使用虚拟环境，先确认虚拟环境已经激活。

### FileNotFoundError

含义：代码要读取的文件不存在，或者路径不对。

排查：

- 文件是否真的存在。
- 当前目录是否正确。
- 文件名是否写错。
- 路径里是否有空格或中文。

可以先打印当前目录：

```python
import os

print(os.getcwd())
```

### 中文乱码

读写文本文件时建议指定编码：

```python
with open("notes.txt", "r", encoding="utf-8") as file:
    content = file.read()
```

保存文件时也尽量使用 UTF-8。

## Linux 常见问题

对应教程：

- [Linux 新手地图](../Linux/docs/part0-newbie-map.md)
- [命令速查附录](../Linux/docs/appendix/command-reference.md)

### command not found

含义：系统找不到这个命令。

可能原因：

- 命令拼错。
- 软件没有安装。
- 当前用户环境变量没有配置。

先检查拼写，再确认是否安装。

### Permission denied

含义：权限不足。

排查：

- 当前用户是否有权限。
- 文件是否可执行。
- 是否需要 `sudo`。

查看权限：

```bash
ls -l 文件名
```

给脚本添加执行权限：

```bash
chmod +x script.sh
```

不要看到权限错误就立刻乱加 `sudo`。先确认命令做什么，尤其是删除、覆盖、安装类命令。

### No such file or directory

含义：路径不存在。

排查：

```bash
pwd
ls
```

确认你所在目录和文件名是否正确。

## Markdown 常见问题

对应教程：

- [Markdown 用法教程](../Markdown/README.md)

### 链接点不开

排查：

- 路径是否写对。
- 文件名大小写是否一致。
- 路径中有空格时是否用了尖括号。
- 目标文件是否已经提交到仓库。

示例：

```markdown
[Matlab 神经网络](<../Matlab/Neural Network/README.md>)
```

### 图片不显示

排查：

- 图片文件是否存在。
- 图片路径是否相对当前 Markdown 文件。
- 图片是否提交到了仓库。
- 文件名大小写是否一致。

### 表格显示错乱

排查：

- 表头和分隔行列数是否一致。
- 每一行是否都有相同数量的 `|`。
- 表格前后是否有空行。

## Matlab 常见问题

对应教程：

- [Matlab 示例运行指南](../Matlab/RUNNING_EXAMPLES.md)
- [Matlab 示例运行索引](../Matlab/EXAMPLE_RUN_INDEX.md)

### Undefined function or variable

常见原因：

- 当前文件夹不对。
- 函数文件不在路径中。
- 文件名和函数名不一致。
- 前置脚本没有运行。

先确认 Current Folder 是否是案例目录。

### Unable to read file

常见原因：

- `.mat`、`.xlsx`、图片文件被移动。
- 文件路径写错。
- 当前工作目录不对。

建议先恢复原目录结构，再运行默认示例。

### 工具箱缺失

神经网络、统计建模、优化算法可能需要不同工具箱。

常见工具箱：

- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- Optimization Toolbox
- Parallel Computing Toolbox

如果报错提示某函数不存在，除了路径问题，也要考虑是否缺少工具箱。

## 问问题时的模板

如果你要向别人求助，可以按这个格式描述：

```markdown
我想做什么：

我运行的命令：

当前目录：

完整报错：

我已经尝试过：
```

这样的描述会比“为什么不行”更容易得到有效回答。

