# Linux 完全新手地图

如果你第一次接触 Linux，不建议一上来就背命令。先理解几个基础概念，后面的命令才不会像一堆陌生咒语。

这一章回答这些问题：

- 终端是什么？
- 路径是什么？
- root 是谁？
- 为什么有权限？
- 服务是什么？
- 为什么教程里总是出现 `sudo`？

## 终端是什么

终端是你和 Linux 对话的地方。

你输入命令：

```bash
pwd
```

Linux 返回结果：

```text
/home/user
```

可以把终端理解成“文字版操作界面”。图形界面里你用鼠标点文件夹，终端里你用命令进入目录、查看文件、运行程序。

## 命令由什么组成

一个命令通常长这样：

```bash
ls -lah /var/log
```

拆开看：

- `ls`：命令名称，表示列出文件
- `-lah`：选项，控制显示方式
- `/var/log`：参数，告诉命令要看哪个目录

不是所有命令都有选项和参数。

例如：

```bash
pwd
```

只需要命令名。

## 路径是什么

路径就是文件或目录的位置。

Linux 路径用 `/` 分隔：

```text
/home/user/readme.txt
```

常见目录：

```text
/             根目录
/home         普通用户目录
/root         root 用户目录
/etc          配置文件
/var/log      日志文件
/tmp          临时文件
```

## 绝对路径和相对路径

绝对路径从 `/` 开始：

```text
/home/user/file.txt
```

相对路径从当前目录开始：

```text
file.txt
./file.txt
../file.txt
```

含义：

- `.` 当前目录
- `..` 上一级目录
- `~` 当前用户的家目录

查看当前目录：

```bash
pwd
```

## 当前目录为什么重要

很多命令会基于当前目录执行。

例如：

```bash
rm file.txt
```

删除的是当前目录下的 `file.txt`。

如果你当前目录不对，就可能删错文件。所以执行危险命令前，先看：

```bash
pwd
ls -lah
```

## 用户和 root

Linux 是多用户系统。

查看当前用户：

```bash
whoami
```

`root` 是最高权限用户，几乎什么都能改。

root 很强，也很危险。用 root 误删系统文件，系统可能直接坏掉。

## sudo 是什么

`sudo` 表示“临时用管理员权限执行命令”。

例如：

```bash
sudo apt update
```

安装软件、修改系统配置、重启服务时，经常需要 `sudo`。

但不要看到命令失败就随手加 `sudo`。先弄清楚它要做什么。

## 权限是什么

Linux 文件有读、写、执行权限。

查看权限：

```bash
ls -l
```

你会看到类似：

```text
-rw-r--r-- 1 user user 1234 Jun 17 file.txt
```

常见权限：

- `r` read，读取
- `w` write，写入
- `x` execute，执行

脚本不能运行时，可能需要：

```bash
chmod +x script.sh
```

但不要随便：

```bash
chmod 777 file
```

`777` 表示所有人都能读、写、执行，风险很高。

## 进程是什么

进程就是正在运行的程序。

查看进程：

```bash
ps aux
top
```

结束进程：

```bash
kill PID
```

`PID` 是进程编号。

## 服务是什么

服务是长期在后台运行的程序。

例如：

- SSH 服务：让你远程登录服务器
- Nginx 服务：提供网站访问
- Docker 服务：运行容器

查看服务状态：

```bash
sudo systemctl status ssh
```

重启服务：

```bash
sudo systemctl restart ssh
```

修改 SSH、防火墙、网络相关服务时要特别小心，因为可能导致远程连接断开。

## 包管理器是什么

包管理器用来安装和更新软件。

Ubuntu / Debian 常用：

```bash
sudo apt update
sudo apt install nginx
```

CentOS / Fedora 可能使用：

```bash
sudo yum install nginx
sudo dnf install nginx
```

不同系统命令不完全一样，先确认自己的发行版：

```bash
cat /etc/os-release
```

## 日志是什么

日志是系统和程序留下的运行记录。

查看 SSH 日志：

```bash
journalctl -u ssh
```

查看某个服务实时日志：

```bash
journalctl -u 服务名 -f
```

排查问题时，日志比猜测可靠。

## 新手安全口诀

执行命令前，先问自己：

1. 我现在在哪个目录？`pwd`
2. 这个命令会不会删除、覆盖、重启、断网？
3. 我有没有备份？
4. 我能不能看懂这个命令的大概意思？
5. 这是官方命令，还是来源不明的一键脚本？

遇到这些命令要特别谨慎：

```bash
rm -rf
chmod 777
dd
mkfs
reboot
shutdown
curl ... | bash
wget ... | bash
```

## 学习顺序建议

建议按这个顺序学：

1. 路径：`pwd`、`cd`、`ls`
2. 文件：`mkdir`、`cp`、`mv`、`rm`
3. 查看：`cat`、`less`、`head`、`tail`
4. 权限：`chmod`、`chown`、`sudo`
5. 进程和服务：`ps`、`top`、`systemctl`
6. 网络：`ping`、`curl`、`ssh`、`scp`
7. 日志：`journalctl`

理解这些概念后，再看后面的命令教程会轻松很多。

