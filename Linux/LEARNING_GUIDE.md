# Linux 学习导读

Linux 新手最容易被大量命令吓住。更有效的学习方式是先建立地图：文件在哪里、权限怎么看、服务怎么查、网络怎么测、危险操作怎么避开。

## 适合人群

- 第一次接触终端和 VPS 的新手。
- 会复制命令，但不清楚命令含义的人。
- 想能独立排查服务器常见问题的人。

## 学习目标

学完后，你应该能：

- 在终端中定位当前目录、查看文件、编辑配置。
- 理解权限、用户、进程、服务、端口之间的关系。
- 用 `systemctl` 和 `journalctl` 初步排查服务问题。
- 识别高风险命令，避免误删和误覆盖。

## 学习地图

```text
文件系统
-> 文件查看和编辑
-> 权限与用户
-> 进程和服务
-> 磁盘、内存、网络
-> SSH 登录与安全
-> VPS 初始化和日常维护
```

## 推荐阅读顺序

1. [Linux 完全新手地图](docs/part0-newbie-map.md)
2. [Linux 是什么，先学什么](docs/part1-basics/1.1-linux-intro.md)
3. [文件和目录操作](docs/part1-basics/1.2-files-and-directories.md)
4. [文件查看、编辑与搜索](docs/part1-basics/1.3-file-view-edit.md)
5. [权限与所有者](docs/part1-basics/1.4-permissions.md)
6. [进程、服务与系统资源](docs/part2-system-admin/2.1-process-service.md)
7. [磁盘、内存和系统信息](docs/part2-system-admin/2.2-disk-memory.md)
8. [软件包管理](docs/part2-system-admin/2.3-package-manager.md)
9. [网络工具与 SSH 登录](docs/part3-network-ssh/3.1-network-tools.md)
10. [SSH 安全加固](docs/part3-network-ssh/3.2-ssh-security.md)

## 核心概念

### 路径

路径分为绝对路径和相对路径。绝对路径从 `/` 开始，例如 `/etc/ssh/sshd_config`。相对路径从当前目录开始，例如 `./logs/app.log`。

新手每次执行删除、移动、复制命令前，先运行：

```bash
pwd
ls -lah
```

确认自己站在哪里，再动手。

### 权限

`rwx` 分别代表读、写、执行。权限不是越大越好，服务器上随手 `chmod 777` 往往会制造安全风险。

常见理解：

- 文件有 `x` 才能作为程序执行。
- 目录有 `x` 才能进入。
- 目录有 `w` 才能在里面创建或删除文件。

### 服务

很多服务器程序不是直接在当前终端里运行，而是被 systemd 管理。常用命令：

```bash
systemctl status nginx
journalctl -u nginx --since "1 hour ago"
```

`systemctl` 看服务状态，`journalctl` 看服务日志。

## 典型场景

### 场景一：VPS 初始化

建议顺序：

1. 更新软件源和系统包。
2. 创建普通用户。
3. 配置 SSH 密钥登录。
4. 禁止密码登录或至少设置强密码。
5. 配置防火墙。
6. 记录系统版本、开放端口、安装服务。

可以对照 [VPS 操作安全清单](docs/part4-vps-tools/4.2-vps-safety-checklist.md)。

### 场景二：服务启动失败

排查顺序：

1. `systemctl status 服务名` 看是否正在运行。
2. `journalctl -u 服务名 -n 100` 看最近日志。
3. 检查配置文件路径是否正确。
4. 检查端口是否被占用。
5. 检查权限和用户是否正确。

不要一开始就重装服务。大多数问题在日志里已经写了原因。

### 场景三：危险命令避坑

执行下面这类命令前要格外小心：

```bash
rm -rf
mv source target
chmod -R
chown -R
dd if=... of=...
```

建议先用 `ls` 确认目标，再在测试目录练习。

## 少量自查

- 你能解释 `/home/user` 和 `./home/user` 的区别吗？
- 你知道 `chmod 777` 为什么危险吗？
- 一个服务无法启动时，你会先看状态还是先重装？
- SSH 登录失败时，你会检查哪些位置？

## 外部资源

- [Linux Journey](https://linuxjourney.com/)：面向新手的 Linux 学习路线。
- [The Debian Administrator's Handbook](https://www.debian.org/doc/manuals/debian-handbook/)：系统管理参考。
- [Ubuntu Server Documentation](https://documentation.ubuntu.com/server/)：Ubuntu Server 官方文档。
- [Arch Wiki](https://wiki.archlinux.org/)：内容非常详细，适合查概念和排错。
- [explainshell](https://explainshell.com/)：把复杂 shell 命令拆开解释。

## 下一步

学完基础命令后，不要急着堆脚本。先把“路径、权限、服务、日志、网络”这五个关键词搞清楚，再去看 VPS 脚本和服务部署。
