# 附录 C：SSH 日志案例与残留进程处理

这个附录整理旧版 `SSH.md` 中的日志案例、分析结论和处理步骤。

## 查看 SSH 日志

Debian / Ubuntu：

```bash
journalctl -u ssh
```

CentOS / RHEL：

```bash
sudo tail -f /var/log/secure
```

## 典型日志片段

```text
Starting ssh.service - OpenBSD Secure Shell server...
Server listening on 0.0.0.0 port 22.
Server listening on :: port 22.
Received signal 15; terminating.
ssh.service: Deactivated successfully.
ssh.service: Unit process 527 (sshd) remains running after unit stopped.
ssh.service: Found left-over process 527 (sshd) in control group while starting unit.
Invalid user AlfaB from 180.167.xxx.xxx port 36219
Disconnected from invalid user AlfaB 180.167.xxx.xxx port 36219 [preauth]
Reloading ssh.service - OpenBSD Secure Shell server...
Received SIGHUP; restarting.
```

## 日志含义

- `Server listening on 0.0.0.0 port 22`: SSH 正在监听 IPv4 的 22 端口。
- `Server listening on :: port 22`: SSH 正在监听 IPv6 的 22 端口。
- `Received signal 15; terminating`: 服务收到终止信号，正在停止。
- `left-over process`: 上一次停止服务后仍有残留 `sshd` 子进程。
- `Invalid user`: 有人尝试使用不存在的用户名登录。
- `[preauth]`: 认证前阶段就被断开，通常出现在爆破或扫描日志中。
- `Received SIGHUP; restarting`: SSH 收到重新加载信号，开始重启或重载配置。

## 排查残留 sshd 进程

先查看服务状态：

```bash
sudo systemctl status ssh
```

如果服务名是 `sshd`：

```bash
sudo systemctl status sshd
```

查看进程：

```bash
ps aux | grep sshd
```

终止单个确认无用的残留进程：

```bash
sudo kill <PID>
```

旧笔记中也记录过：

```bash
sudo killall sshd
```

但这条命令风险更高，可能断开当前 SSH 连接。新手不建议在远程服务器上直接执行，除非你已经有控制台、VNC 或其他登录方式兜底。

## 检查 ssh.service 配置

查看状态：

```bash
sudo systemctl status ssh
```

编辑 systemd 覆盖配置：

```bash
sudo systemctl edit ssh
```

旧笔记中提到的配置：

```ini
[Service]
KillMode=process
```

修改后重新加载 systemd 并重启：

```bash
sudo systemctl daemon-reload
sudo systemctl restart ssh
```

注意：`KillMode` 涉及 systemd 如何终止服务进程。不了解影响时，不建议随意修改生产服务器配置。

## SSH 安全设置

限制允许登录的用户：

```ini
AllowUsers your_username
```

启用密钥认证并禁用密码登录：

```ini
PasswordAuthentication no
PubkeyAuthentication yes
```

使用防火墙限制来源 IP：

```bash
sudo ufw allow from 192.168.1.100 to any port 22
```

修改 SSH 配置前，建议先阅读 [SSH 安全加固与日志排查](../part3-network-ssh/3.2-ssh-security.md)，避免把自己锁在服务器外面。

