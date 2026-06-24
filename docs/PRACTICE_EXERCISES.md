# 新手练习题与自查答案

这份练习用于配合仓库里的基础教程。它的目标不是考试，而是帮你确认自己是否真的会操作。

建议做法：

1. 先读对应教程。
2. 不看答案自己做一遍。
3. 做完后再对照自查答案。
4. 如果卡住，回到对应教程复习。

## Python 基础练习

对应教程：

- [Python 基础教程](../Python/Basics/README.md)

### 练习 1：打印个人信息

任务：

写一个 `profile.py`，输出你的姓名、学习目标和今天要完成的一件事。

参考代码：

```python
name = "小明"
goal = "学会运行 Python 脚本"
todo = "完成第一个练习"

print("姓名：", name)
print("目标：", goal)
print("今天要做：", todo)
```

自查：

- 文件是否以 `.py` 结尾。
- 是否能用 `python profile.py` 运行。
- 输出内容是否和你设置的变量一致。

### 练习 2：输入两个数字并求和

任务：

让用户输入两个数字，程序输出它们的和。

参考代码：

```python
a = float(input("请输入第一个数字："))
b = float(input("请输入第二个数字："))

print("相加结果：", a + b)
```

自查：

- `input()` 得到的是字符串，是否转换成了数字。
- 输入小数时是否也能正常计算。
- 如果输入文字，基础版本会报错；可以继续看下面的异常处理版本。

进阶参考代码：

```python
try:
    a = float(input("请输入第一个数字："))
    b = float(input("请输入第二个数字："))
except ValueError:
    print("输入错误：请输入数字。")
else:
    print("相加结果：", a + b)
```

### 练习 3：统计列表中的偶数

任务：

给定一个数字列表，统计里面有多少个偶数。

参考代码：

```python
numbers = [1, 2, 3, 4, 5, 6, 10, 13]
count = 0

for number in numbers:
    if number % 2 == 0:
        count += 1

print("偶数数量：", count)
```

自查：

- 是否使用了 `for` 循环。
- 是否使用 `% 2 == 0` 判断偶数。
- 输出结果应该是 `4`。

### 练习 4：读取文本文件

任务：

创建一个 `notes.txt`，写入几行文字。再写一个 Python 脚本读取并打印它。

参考代码：

```python
with open("notes.txt", "r", encoding="utf-8") as file:
    content = file.read()

print(content)
```

自查：

- `notes.txt` 是否和脚本在同一个目录。
- 是否指定了 `encoding="utf-8"`。
- 文件不存在时，基础版本会报错；可以继续看下面的异常处理版本。

进阶参考代码：

```python
try:
    with open("notes.txt", "r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    print("没有找到 notes.txt，请确认文件和脚本在同一个目录。")
else:
    print(content)
```

## Linux 基础练习

对应教程：

- [Linux 新手地图](../Linux/docs/part0-newbie-map.md)
- [文件和目录操作](../Linux/docs/part1-basics/1.2-files-and-directories.md)

### 练习 1：创建练习目录

任务：

在当前目录下创建 `learn-linux` 目录，并进入它。

参考命令：

```bash
mkdir learn-linux
cd learn-linux
pwd
```

自查：

- `pwd` 显示的路径末尾是否是 `learn-linux`。
- 如果目录已存在，`mkdir` 可能提示已存在，这不是大问题。

### 练习 2：创建并查看文件

任务：

创建一个 `hello.txt`，写入一行文字，然后查看内容。

参考命令：

```bash
echo "hello linux" > hello.txt
cat hello.txt
```

自查：

- `cat hello.txt` 是否输出 `hello linux`。
- `>` 会覆盖文件内容，追加内容应使用 `>>`。

### 练习 3：复制、重命名、删除

任务：

复制 `hello.txt`，再把副本改名，最后删除它。

参考命令：

```bash
cp hello.txt hello-copy.txt
mv hello-copy.txt backup.txt
rm backup.txt
```

自查：

- `cp` 后是否出现了副本。
- `mv` 后旧文件名是否消失。
- `rm` 删除前是否确认文件名正确。

### 练习 4：查看最近日志

任务：

创建一个多行文件，然后查看前几行和后几行。

参考命令：

```bash
printf "line1\nline2\nline3\nline4\nline5\n" > log.txt
head -n 2 log.txt
tail -n 2 log.txt
```

自查：

- `head -n 2` 是否显示前 2 行。
- `tail -n 2` 是否显示后 2 行。

## GitHub / Git 基础练习

对应教程：

- [GitHub 新手入门教程](../Github/README.md)
- [第一次提交练习](../Github/docs/part2-git-cmds/2.5-first-commit-practice.md)

### 练习 1：查看仓库状态

任务：

在仓库根目录运行状态检查命令。

参考命令：

```bash
git status
```

自查：

- 如果显示 `working tree clean`，说明当前没有未提交改动。
- 如果显示红色文件，说明这些文件还没有暂存。
- 如果显示绿色文件，说明这些文件已经暂存。

### 练习 2：完成一次文档提交

任务：

修改一篇 Markdown 文件，完成一次提交。

参考命令：

```bash
git status
git add README.md
git commit -m "docs: update readme"
```

自查：

- 提交信息是否能说明你改了什么。
- 不要所有提交都写 `update`。
- 提交后再次运行 `git status`，确认是否干净。

### 练习 3：推送到 GitHub

任务：

把本地提交推送到远程仓库。

参考命令：

```bash
git push origin main
```

自查：

- GitHub 网页上是否能看到最新提交。
- 如果提示登录失败，检查 Git 凭据或 SSH 配置。
- 如果提示远程分支比本地新，先了解 `git pull`，不要盲目覆盖。

### 练习 4：看懂一次 diff

任务：

修改文件后，在提交前查看差异。

参考命令：

```bash
git diff
```

自查：

- `-` 开头通常表示删除的内容。
- `+` 开头通常表示新增的内容。
- 提交前看 diff 能减少误提交。

## 自查清单

完成这些练习后，你应该能做到：

- 写并运行一个简单 Python 脚本。
- 在命令行中创建、查看、移动、删除测试文件。
- 用 Git 查看状态、提交修改、推送到 GitHub。
- 看懂最基础的错误提示，并知道先检查路径、文件名和依赖。

如果上面任何一步还不熟，建议不要急着学更复杂内容。基础操作越稳，后面学脚本、服务器和算法时越轻松。
