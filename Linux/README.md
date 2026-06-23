# Linux 新手入门与 VPS 实用手册

这个目录整理 Linux 基础命令、SSH 安全、VPS 常用脚本与排障经验。它的目标不是堆命令，而是让新手知道：什么时候用、为什么用、哪里危险。

## 推荐阅读路线

1. [Linux 学习导读](LEARNING_GUIDE.md)
2. [Linux 完全新手地图](docs/part0-newbie-map.md)
3. [Linux 是什么，先学什么](docs/part1-basics/1.1-linux-intro.md)
4. [文件和目录操作](docs/part1-basics/1.2-files-and-directories.md)
5. [文件查看、编辑与搜索](docs/part1-basics/1.3-file-view-edit.md)
6. [权限与所有者](docs/part1-basics/1.4-permissions.md)
7. [进程、服务与系统资源](docs/part2-system-admin/2.1-process-service.md)
8. [磁盘、内存和系统信息](docs/part2-system-admin/2.2-disk-memory.md)
9. [软件包管理](docs/part2-system-admin/2.3-package-manager.md)
10. [网络工具与 SSH 登录](docs/part3-network-ssh/3.1-network-tools.md)
11. [SSH 安全加固](docs/part3-network-ssh/3.2-ssh-security.md)
12. [VPS 常用脚本速查](docs/part4-vps-tools/4.1-vps-script-index.md)
13. [旧内容迁移索引](docs/appendix/original-content-map.md)

## 内容地图

### 基础命令

- [Linux 学习导读](LEARNING_GUIDE.md)
- [Linux 常见问题](FAQ.md)
- [Linux 完全新手地图](docs/part0-newbie-map.md)
- [Linux 入门概念](docs/part1-basics/1.1-linux-intro.md)
- [文件和目录操作](docs/part1-basics/1.2-files-and-directories.md)
- [文件查看、编辑与搜索](docs/part1-basics/1.3-file-view-edit.md)
- [权限与所有者](docs/part1-basics/1.4-permissions.md)

### 系统管理

- [进程、服务与后台任务](docs/part2-system-admin/2.1-process-service.md)
- [磁盘、内存和系统信息](docs/part2-system-admin/2.2-disk-memory.md)
- [软件包管理](docs/part2-system-admin/2.3-package-manager.md)

### 网络与 SSH

- [网络工具与远程传输](docs/part3-network-ssh/3.1-network-tools.md)
- [SSH 安全加固与日志排查](docs/part3-network-ssh/3.2-ssh-security.md)

### VPS 工具

- [VPS 常用脚本速查表](docs/part4-vps-tools/4.1-vps-script-index.md)
- [VPS 操作安全清单](docs/part4-vps-tools/4.2-vps-safety-checklist.md)

### 附录

- [旧内容迁移索引](docs/appendix/original-content-map.md)
- [命令速查补全表](docs/appendix/command-reference.md)
- [SSH 日志案例与残留进程处理](docs/appendix/ssh-log-case.md)

## 删除旧文档前看这里

旧内容已经整理进 `docs/`：

- `LinuxCMD_Basics.md` 的命令内容已进入基础、系统管理、网络章节，以及 [命令速查补全表](docs/appendix/command-reference.md)。
- `SSH.md` 的日志分析和安全建议已进入 [SSH 安全加固与日志排查](docs/part3-network-ssh/3.2-ssh-security.md) 和 [SSH 日志案例与残留进程处理](docs/appendix/ssh-log-case.md)。
- `VPSscript.md` 的脚本命令已进入 [VPS 常用脚本速查表](docs/part4-vps-tools/4.1-vps-script-index.md)。

如果你想确认旧内容的迁移位置，看 [旧内容迁移索引](docs/appendix/original-content-map.md)。

整理前的三篇原始笔记分别是 `LinuxCMD_Basics.md`、`SSH.md`、`VPSscript.md`。它们的主要命令、脚本、日志案例和来源链接已经迁移到 `docs/`，删除旧文件后仍可通过上面的迁移索引查找。
