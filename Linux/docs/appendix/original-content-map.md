# 附录 A：旧内容迁移索引

这个索引用来回答一个问题：删除整理前的旧文档后，原来的内容到哪里找？

## LinuxCMD_Basics.md 迁移位置

| 旧内容 | 新位置 |
| :--- | :--- |
| 博客原文链接 `https://blog.luopojunzi.com/p/linuxCMD/` | 本页“原始来源链接” |
| 文件和目录操作：`ls`、`cd`、`pwd`、`mkdir`、`rmdir`、`rm`、`cp`、`mv`、`touch`、`find` | [文件和目录操作](../part1-basics/1.2-files-and-directories.md) |
| 文件查看：`cat`、`more`、`less`、`head`、`tail` | [文件查看、编辑与搜索](../part1-basics/1.3-file-view-edit.md) |
| 文件编辑：`nano`、`vi`、`vim` | [文件查看、编辑与搜索](../part1-basics/1.3-file-view-edit.md) 和 [命令速查补全表](command-reference.md) |
| 文本搜索处理：`grep`、`awk`、`sed` | [文件查看、编辑与搜索](../part1-basics/1.3-file-view-edit.md) |
| 权限：`chmod`、`chown`、`sudo` | [权限与所有者](../part1-basics/1.4-permissions.md) |
| 系统管理：`ps`、`top`、`kill` | [进程、服务与后台任务](../part2-system-admin/2.1-process-service.md) |
| 磁盘内存：`df`、`du`、`free` | [磁盘、内存和系统信息](../part2-system-admin/2.2-disk-memory.md) |
| 网络操作：`ping`、`ifconfig`、`ssh`、`scp`、`wget` | [网络工具与远程传输](../part3-network-ssh/3.1-network-tools.md) |
| 软件包管理：`apt`、`yum`、`dnf`、`pacman` | [软件包管理](../part2-system-admin/2.3-package-manager.md) |
| 压缩与解压：`tar`、`zip`、`unzip` | [文件和目录操作](../part1-basics/1.2-files-and-directories.md) |
| 系统信息：`uname`、`hostname`、`whoami`、`id`、`dmesg`、`lsb_release`、`uptime` | [磁盘、内存和系统信息](../part2-system-admin/2.2-disk-memory.md) |
| 进程管理：`bg`、`fg`、`jobs`、`killall`、`pkill`、`htop`、`nice`、`renice`、`nohup`、`strace` | [进程、服务与后台任务](../part2-system-admin/2.1-process-service.md) 和 [命令速查补全表](command-reference.md) |
| VI 编辑器操作命令 | [命令速查补全表](command-reference.md) |

## SSH.md 迁移位置

| 旧内容 | 新位置 |
| :--- | :--- |
| 查看 CentOS `/var/log/secure` 和 Debian/Ubuntu `journalctl -u ssh` | [SSH 安全加固与日志排查](../part3-network-ssh/3.2-ssh-security.md) |
| `Invalid user`、爆破尝试日志含义 | [SSH 安全加固与日志排查](../part3-network-ssh/3.2-ssh-security.md) |
| `ssh.service` 启停、残留 `sshd` 进程分析 | [SSH 日志案例与残留进程处理](ssh-log-case.md) |
| 查找残留 `sshd` 进程：`ps aux \| grep sshd` | [SSH 日志案例与残留进程处理](ssh-log-case.md) |
| 终止进程：`sudo kill <PID>`、`sudo killall sshd` | [SSH 日志案例与残留进程处理](ssh-log-case.md)，并补充了风险提示 |
| 检查服务状态：`sudo systemctl status ssh` | [进程、服务与后台任务](../part2-system-admin/2.1-process-service.md) 和 [SSH 日志案例与残留进程处理](ssh-log-case.md) |
| `sudo systemctl edit ssh`、`KillMode=process` | [SSH 日志案例与残留进程处理](ssh-log-case.md) |
| `AllowUsers`、`PasswordAuthentication no`、`PubkeyAuthentication yes` | [SSH 安全加固与日志排查](../part3-network-ssh/3.2-ssh-security.md) |
| `ufw allow from ... to any port 22` | [SSH 安全加固与日志排查](../part3-network-ssh/3.2-ssh-security.md) |

## VPSscript.md 迁移位置

| 旧内容 | 新位置 |
| :--- | :--- |
| DD 重装脚本、MoeClub 脚本、DD Windows | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) |
| 融合怪、NodeBench、yabs、GB5 | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) |
| 流媒体、IP 质量、ChatGPT App 解锁检测 | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) |
| 三网测速、全球测速、回程测试 | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) |
| Fail2ban、BBR、TCP 调优、SWAP、25 端口测试 | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) |
| Docker、Python、WARP、Aria2、aaPanel、宝塔 | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) |
| 科技 lion、SKY-BOX | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) |
| 原帖链接 `https://www.nodeseek.com/post-143131-1` | [VPS 常用脚本速查表](../part4-vps-tools/4.1-vps-script-index.md) 的“原始来源” |
| 原“关注我”链接 | 本页“原始来源链接” |

## 原始来源链接

- Linux 命令博客原文：`https://blog.luopojunzi.com/p/linuxCMD/`
- VPS 脚本论坛原帖：`https://www.nodeseek.com/post-143131-1`
- 原笔记作者主页：`https://www.nodeseek.com/space/18714#/general`

