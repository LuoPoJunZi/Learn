# SSH安全
可查看 CentOS 的 /var/log/secure 文件或者 Debian/Ubuntu 的 journalctl -u ssh 的内容，这里都是一些尝试爆破你SSH密码的风险IP
```
journalctl -u ssh
```
运行结果如下（声明：我已经关闭了22端口的密码登录）：
```
Nov 04 09:33:50 s2575 systemd[1]: Starting ssh.service - OpenBSD Secure Shell server...
Nov 04 09:33:50 s2575 sshd[525]: Server listening on 0.0.0.0 port 22.
Nov 04 09:33:50 s2575 sshd[525]: Server listening on :: port 22.
Nov 04 09:33:50 s2575 systemd[1]: Started ssh.service - OpenBSD Secure Shell server.
Nov 04 09:33:50 s2575 sshd[525]: Received signal 15; terminating.
Nov 04 09:33:50 s2575 systemd[1]: Stopping ssh.service - OpenBSD Secure Shell server...
Nov 04 09:33:50 s2575 systemd[1]: ssh.service: Deactivated successfully.
Nov 04 09:33:50 s2575 systemd[1]: ssh.service: Unit process 527 (sshd) remains running after unit stopped.
Nov 04 09:33:50 s2575 systemd[1]: ssh.service: Unit process 528 (sshd) remains running after unit stopped.
Nov 04 09:33:50 s2575 systemd[1]: Stopped ssh.service - OpenBSD Secure Shell server.
Nov 04 09:33:50 s2575 systemd[1]: ssh.service: Found left-over process 527 (sshd) in control group while starting uni>
Nov 04 09:33:50 s2575 systemd[1]: This usually indicates unclean termination of a previous run, or service implementa>
Nov 04 09:33:50 s2575 systemd[1]: ssh.service: Found left-over process 528 (sshd) in control group while starting uni>
Nov 04 09:33:50 s2575 systemd[1]: This usually indicates unclean termination of a previous run, or service implementa>
Nov 04 09:33:50 s2575 systemd[1]: Starting ssh.service - OpenBSD Secure Shell server...
Nov 04 09:33:50 s2575 systemd[1]: ssh.service: Found left-over process 527 (sshd) in control group while starting uni>
Nov 04 09:33:50 s2575 systemd[1]: This usually indicates unclean termination of a previous run, or service implementa>
Nov 04 09:33:50 s2575 systemd[1]: ssh.service: Found left-over process 528 (sshd) in control group while starting uni>
Nov 04 09:33:50 s2575 systemd[1]: This usually indicates unclean termination of a previous run, or service implementa>
Nov 04 09:33:50 s2575 sshd[546]: Server listening on 0.0.0.0 port 22.
Nov 04 09:33:50 s2575 sshd[546]: Server listening on :: port 22.
Nov 04 09:33:50 s2575 systemd[1]: Started ssh.service - OpenBSD Secure Shell server.
Nov 04 09:33:50 s2575 sshd[527]: Invalid user AlfaB from 180.167.xxx.xxx port 36219 #IP已经隐藏
Nov 04 09:33:50 s2575 sshd[527]: Received disconnect from 180.167.xxx.xxx port 36219:11: Bye Bye [preauth] #IP已经隐藏
Nov 04 09:33:50 s2575 sshd[527]: Disconnected from invalid user AlfaB 180.167.xxx.xxx port 36219 [preauth] #IP已经隐藏
Nov 04 09:33:51 s2575 systemd[1]: Reloading ssh.service - OpenBSD Secure Shell server...
Nov 04 09:33:51 s2575 systemd[1]: Reloaded ssh.service - OpenBSD Secure Shell server.
Nov 04 09:33:51 s2575 sshd[546]: Received SIGHUP; restarting.
Nov 04 09:33:51 s2575 sshd[546]: Server listening on 0.0.0.0 port 22.
```
## ChatGPT回应
从日志来看，SSH 服务 (`ssh.service`) 似乎在短时间内经历了几次启动和停止的过程，并且存在一些未正常终止的子进程。这些日志信息可以总结如下：
1. **SSH 服务启动**：
   - `Starting ssh.service - OpenBSD Secure Shell server...`
   - 服务开始监听 `0.0.0.0` 和 `::` 的端口 22，表明 SSH 服务器启动成功并正在监听 IPv4 和 IPv6。

2. **SSH 服务停止和残留进程**：
   - 收到 `Received signal 15; terminating` 表明 SSH 服务收到了终止信号（通常是 `SIGTERM`），因此服务尝试关闭。
   - 尽管 `ssh.service` 成功停用（`ssh.service: Deactivated successfully`），但仍然有两个子进程（`sshd`）未完全关闭，并且在再次启动时被系统识别为“残留”进程。
   - 日志中的“left-over process”表示这些进程在前一次运行后未被正确清理，可能是因为服务实现存在问题或异常终止。

3. **无效登录尝试**：
   - 日志中显示了一次无效的 SSH 登录尝试：`Invalid user AlfaB from 180.167.153.230`。
   - 远程主机 `180.167.153.230` 试图使用用户名 `AlfaB` 登录，但由于用户名无效，被服务器拒绝连接。

4. **服务重新加载**：
   - `Reloading ssh.service - OpenBSD Secure Shell server...` 表明 SSH 服务正在重新加载配置。
   - `Received SIGHUP; restarting` 表示 SSH 服务收到了 `SIGHUP` 信号，导致其重启并重新开始监听。

## 建议
- **清理残留进程**：可以通过手动杀死这些残留的 `sshd` 进程，确保服务停止时清理所有子进程。
- **检查配置文件和服务状态**：可能需要检查 `ssh.service` 的配置，确保其能正常停止所有子进程。
- **安全设置**：如果无效的 SSH 尝试频繁发生，建议加强 SSH 安全设置，例如禁用无效用户名登录、使用强密码或密钥认证，并考虑使用防火墙限制访问。

通过以下步骤执行建议来清理残留进程、检查配置文件，以及加强 SSH 安全设置：

### 1. 清理残留的 `sshd` 进程
首先，找出并手动终止残留的 `sshd` 进程，确保服务干净地重新启动。

#### 执行步骤
1. **查找残留的 `sshd` 进程**：
   ```bash
   ps aux | grep sshd
   ```
   找到与 `sshd` 相关的所有进程。

2. **杀死残留的 `sshd` 进程**：
   根据查找到的进程 ID（PID），可以使用以下命令逐个终止它们：
   ```bash
   sudo kill <PID>
   ```
   例如，`sudo kill 527` 将终止 ID 为 527 的进程。如果有多个残留进程，可以重复执行该命令，或者使用 `killall` 一次性结束所有 `sshd` 进程：
   ```bash
   sudo killall sshd
   ```

3. **重新启动 SSH 服务**：
   清理完成后，重新启动 SSH 服务，确保其正常工作：
   ```bash
   sudo systemctl start ssh
   ```

### 2. 检查 `ssh.service` 配置和服务状态
确保 `ssh.service` 在每次停止时能正确清理其所有进程，并正确启动。

#### 执行步骤
1. **查看 SSH 服务状态**：
   检查服务状态是否正常，是否有任何错误或警告：
   ```bash
   sudo systemctl status ssh
   ```

2. **编辑 `ssh.service` 配置（如有必要）**：
   如果 SSH 服务频繁出现“残留进程”问题，可以检查其配置文件：
   ```bash
   sudo systemctl edit ssh
   ```
   这将打开 `ssh.service` 的自定义覆盖配置。如果需要将 `KillMode` 设置为 `process` 以确保所有子进程在停止时都被清理，请添加以下内容：
   ```ini
   [Service]
   KillMode=process
   ```

3. **重启 SSH 服务**：
   编辑完配置后，重新加载 `systemd` 配置并重启服务：
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ssh
   ```

### 3. 加强 SSH 安全设置
由于日志显示有无效登录尝试，建议配置 SSH 以提高安全性。

#### 执行步骤
1. **编辑 SSH 配置文件**：
   打开 SSH 配置文件 `sshd_config`：
   ```bash
   sudo nano /etc/ssh/sshd_config
   ```

2. **禁用无效用户名登录**：
   可以通过限制允许的用户来阻止不在列表中的用户登录。例如，添加以下行来只允许特定用户：
   ```bash
   AllowUsers your_username
   ```
   替换 `your_username` 为你希望允许访问的用户。

3. **启用密钥认证并禁用密码登录**（如适用）：
   密钥认证比密码更加安全，启用密钥认证并禁用密码登录可以提高 SSH 的安全性：
   ```ini
   PasswordAuthentication no
   PubkeyAuthentication yes
   ```

4. **使用防火墙限制访问**：
   使用 `ufw` 或 `iptables` 防火墙，只允许特定 IP 地址访问 SSH 端口 22。例如，使用 `ufw` 来允许特定 IP 地址（如 `192.168.1.100`）访问：
   ```bash
   sudo ufw allow from 192.168.1.100 to any port 22
   ```

5. **重启 SSH 服务**：
   修改 SSH 配置后，重新启动 SSH 服务应用更改：
   ```bash
   sudo systemctl restart ssh
   ```
