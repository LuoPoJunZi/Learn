# 博客[原文](https://blog.luopojunzi.com/p/linuxCMD/)

# 常见的Linux命令

## 1. **文件和目录操作**
```bash
`ls`: 列出目录内容。
`cd`: 切换当前工作目录。
`pwd`: 显示当前工作目录的路径。
`mkdir`: 创建新目录。
`rmdir`: 删除空目录。
`rm`: 删除文件或目录（`rm -r`递归删除目录及其内容）。
`cp`: 复制文件或目录。
`mv`: 移动或重命名文件或目录。
`touch`: 创建一个新的空文件或更新文件的时间戳。
`cat`: 显示文件内容。
`more` / `less`: 分页显示文件内容。
`head`: 显示文件的前几行。
`tail`: 显示文件的最后几行。
`find`: 搜索文件和目录。
`chmod`: 修改文件或目录的权限。
`chown`: 改变文件或目录的所有者。
```
#### `ls`
- **描述**: 列出目录内容。
- **示例**:
  ```bash
  ls            # 列出当前目录的文件和子目录
  ls -l         # 以长格式显示文件详细信息（权限、所有者、大小、修改时间等）
  ls -a         # 列出所有文件，包括隐藏文件（以.开头）
  ls -lh        # 以易读的格式显示文件大小（如KB、MB）
  ```

#### `cd`
- **描述**: 切换当前工作目录。
- **示例**:
  ```bash
  cd /home/user/Documents  # 切换到指定路径
  cd ..                    # 返回到上一级目录
  cd ~                     # 切换到用户的主目录
  ```

#### `pwd`
- **描述**: 显示当前工作目录的路径。
- **示例**:
  ```bash
  pwd  # 输出当前所在的目录路径，例如 /home/user/Documents
  ```

#### `mkdir`
- **描述**: 创建新目录。
- **示例**:
  ```bash
  mkdir new_folder          # 创建名为new_folder的目录
  mkdir -p dir1/dir2/dir3   # 递归创建目录，如果父目录不存在也会一并创建
  ```

#### `rmdir`
- **描述**: 删除空目录。
- **示例**:
  ```bash
  rmdir empty_folder  # 删除名为empty_folder的空目录
  ```

#### `rm`
- **描述**: 删除文件或目录。
- **示例**:
  ```bash
  rm file.txt             # 删除文件file.txt
  rm -r directory         # 递归删除目录及其内容
  rm -rf directory        # 强制递归删除目录及其内容，不提示确认
  ```

#### `cp`
- **描述**: 复制文件或目录。
- **示例**:
  ```bash
  cp file1.txt file2.txt         # 复制文件file1.txt为file2.txt
  cp -r directory1 directory2    # 递归复制目录directory1及其内容到directory2
  ```

#### `mv`
- **描述**: 移动或重命名文件或目录。
- **示例**:
  ```bash
  mv old_name.txt new_name.txt   # 重命名文件old_name.txt为new_name.txt
  mv file.txt /path/to/directory # 移动file.txt到指定目录
  ```

#### `touch`
- **描述**: 创建一个新的空文件或更新文件的时间戳。
- **示例**:
  ```bash
  touch newfile.txt  # 创建一个空文件newfile.txt
  touch existingfile.txt  # 更新文件existingfile.txt的访问和修改时间
  ```

#### `cat`
- **描述**: 显示文件内容，通常用于查看文本文件的内容。
- **示例**:
  ```bash
  cat file.txt               # 显示file.txt的内容
  cat file1.txt file2.txt    # 依次显示file1.txt和file2.txt的内容
  ```

#### `more` / `less`
- **描述**: 分页显示文件内容，便于逐屏阅读。
- **示例**:
  ```bash
  more file.txt  # 分屏查看file.txt的内容，使用空格键翻页
  less file.txt  # 类似more，但更灵活，支持上下翻页
  ```

#### `head`
- **描述**: 显示文件的前几行。
- **示例**:
  ```bash
  head file.txt        # 显示file.txt的前10行
  head -n 20 file.txt  # 显示file.txt的前20行
  ```

#### `tail`
- **描述**: 显示文件的最后几行。
- **示例**:
  ```bash
  tail file.txt        # 显示file.txt的最后10行
  tail -n 20 file.txt  # 显示file.txt的最后20行
  tail -f log.txt      # 实时跟踪显示log.txt的新内容（用于监视日志文件）
  ```

#### `find`
- **描述**: 搜索文件和目录。
- **示例**:
  ```bash
  find /path -name "*.txt"           # 在/path目录及其子目录中搜索所有以.txt结尾的文件
  find /path -type d -name "backup"  # 搜索名称为backup的目录
  ```

#### `chmod`
- **描述**: 修改文件或目录的权限。
- **示例**:
  ```bash
  chmod 755 script.sh  # 设置文件script.sh的权限为rwxr-xr-x
  chmod u+x script.sh  # 仅给予文件所有者执行权限
  ```

#### `chown`
- **描述**: 改变文件或目录的所有者。
- **示例**:
  ```bash
  chown user:group file.txt        # 将文件file.txt的所有者改为user，所属组改为group
  chown -R user:group directory/   # 递归改变目录及其内容的所有者
  ```

## 2. **文件查看和编辑**
```bash
`nano`: 一个简单的文本编辑器。
`vi` / `vim`: 功能强大的文本编辑器。
`grep`: 在文件中搜索文本模式。
`awk`: 文本处理工具，适用于处理和分析文本数据。
`sed`: 流编辑器，用于文本的查找和替换。
```
#### `nano`
- **描述**: 一个简单的命令行文本编辑器。
- **示例**:
  ```bash
  nano file.txt  # 在nano编辑器中打开file.txt进行编辑
  ```

#### `vi` / `vim`
- **描述**: 功能强大的文本编辑器，支持模式编辑。
- **示例**:
  ```bash
  vi file.txt   # 打开file.txt进行编辑
  vim file.txt  # 使用vim打开file.txt，vim是vi的增强版
  ```
  - **基本操作**:
    - `i`: 切换到插入模式以编辑文本。
    - `:wq`: 保存并退出。
    - `:q!`: 不保存退出。

#### `grep`
- **描述**: 在文件中搜索指定的文本模式。
- **示例**:
  ```bash
  grep "hello" file.txt                # 在file.txt中搜索包含"hello"的行
  grep -r "function" /path/to/dir      # 递归搜索目录中的所有文件，寻找包含"function"的行
  grep -i "pattern" file.txt           # 不区分大小写地搜索"pattern"
  ```

#### `awk`
- **描述**: 文本处理工具，适用于从文本文件或命令输出中提取和操作数据。
- **示例**:
  ```bash
  awk '{print $1, $3}' file.txt  # 打印file.txt中每一行的第1和第3列
  ```

#### `sed`
- **描述**: 流编辑器，用于文本的查找和替换。
- **示例**:
  ```bash
  sed 's/oldtext/newtext/g' file.txt  # 将file.txt中所有的oldtext替换为newtext
  sed -n '1,5p' file.txt              # 只显示file.txt的第1到第5行
  ```

## 3. **系统管理**
```bash
`sudo`: 以超级用户（root）权限运行命令。
`ps`: 显示当前运行的进程。
`top`: 动态显示系统的运行情况，包括CPU、内存使用情况。
`kill`: 终止进程（`kill -9`强制终止）。
`df`: 显示磁盘空间使用情况。
`du`: 显示目录或文件的磁盘使用情况。
`free`: 显示内存使用情况。
`uptime`: 显示系统的运行时间和负载平均值。
`shutdown`: 关闭或重启系统。
```
#### `sudo`
- **描述**: 以超级用户（root）权限运行命令。
- **示例**:
  ```bash
  sudo apt-get update  # 以root权限更新软件包列表
  sudo rm -r /var/log  # 以root权限删除/var/log目录
  ```

#### `ps`
- **描述**: 显示当前运行的进程。
- **示例**:
  ```bash
  ps aux                # 显示所有用户的所有进程
  ps -ef                # 以全格式显示当前系统中的进程
  ```

#### `top`
- **描述**: 动态显示系统的运行情况，包括CPU、内存使用情况。
- **示例**:
  ```bash
  top  # 实时显示系统的资源使用情况，按`q`退出
  ```

#### `kill`
- **描述**: 终止进程。
- **示例**:
  ```bash
  kill 1234       # 终止进程ID为1234的进程
  kill -9 1234    # 强制终止进程ID为1234的进程（SIGKILL信号）
  ```

#### `df`
- **描述**: 显示磁盘空间使用情况。
- **示例**:
  ```bash
  df -h           # 以人类可读的格式显示磁盘使用情况（如KB、MB）
  df /dev/sda1    # 显示指定设备的磁盘使用情况
  ```

#### `du`
- **描述**: 显示目录或文件的磁盘使用情况。
- **示例**:
  ```bash
  du -sh /home/user    # 显示/home/user目录的总大小
  du -h --max-depth=1  #

 显示当前目录下每个子目录的大小
  ```

#### `free`
- **描述**: 显示系统的内存使用情况。
- **示例**:
  ```bash
  free -h  # 以人类可读格式显示内存使用情况
  ```

## 4. **网络操作**
```bash
`ping`: 检查网络连接是否正常。
`ifconfig`: 显示或配置网络接口。
`curl`: 命令行工具，用于发送HTTP请求。
`wget`: 用于从网络下载文件。
`ssh`: 通过网络远程登录到另一台计算机。
`scp`: 在本地和远程计算机之间复制文件。
`ftp`: 文件传输协议客户端。
`netstat`: 显示网络连接、路由表和网络接口统计信息。
```
#### `ping`
- **描述**: 测试与主机的网络连通性。
- **示例**:
  ```bash
  ping google.com         # 测试与google.com的连通性
  ping -c 4 192.168.1.1   # 发送4个数据包测试与指定IP的连接
  ```

#### `ifconfig`
- **描述**: 显示或配置网络接口（现代系统使用`ip`命令替代）。
- **示例**:
  ```bash
  ifconfig             # 显示所有网络接口的详细信息
  ifconfig eth0 down   # 关闭eth0网络接口
  ifconfig eth0 up     # 启动eth0网络接口
  ```

#### `ssh`
- **描述**: 使用SSH协议连接到远程主机。
- **示例**:
  ```bash
  ssh user@remote_host  # 使用用户名user连接到远程主机remote_host
  ssh -p 2222 user@host # 使用指定端口2222连接
  ```

#### `scp`
- **描述**: 通过SSH在本地和远程主机之间安全地复制文件。
- **示例**:
  ```bash
  scp file.txt user@remote_host:/path/to/destination  # 将文件复制到远程主机
  scp -r directory user@remote_host:/path/to/destination  # 递归复制目录
  ```

#### `wget`
- **描述**: 从网络下载文件。
- **示例**:
  ```bash
  wget http://example.com/file.txt   # 下载指定URL的文件
  wget -c http://example.com/file.txt  # 断点续传文件
  ```





## 5. **软件包管理**
```bash
`apt-get`: Debian/Ubuntu系统的包管理工具，用于安装、更新、删除软件包。
`yum`: RedHat/CentOS系统的包管理工具，用于管理软件包。
`dnf`: Fedora系统的包管理工具，是`yum`的继任者。
`pacman`: Arch Linux系统的包管理工具。
```
### `apt` (适用于基于Debian的系统，例如Ubuntu)

#### `apt update`
- **描述**: 更新本地包索引。
- **示例**:
  ```bash
  sudo apt update  # 更新可用软件包列表
  ```

#### `apt upgrade`
- **描述**: 升级已安装的软件包。
- **示例**:
  ```bash
  sudo apt upgrade  # 升级系统中所有已安装的软件包到最新版本
  ```

#### `apt install`
- **描述**: 安装新的软件包。
- **示例**:
  ```bash
  sudo apt install package_name  # 安装名为package_name的软件包
  ```

#### `apt remove`
- **描述**: 删除已安装的软件包。
- **示例**:
  ```bash
  sudo apt remove package_name  # 删除名为package_name的软件包
  ```

#### `apt autoremove`
- **描述**: 删除系统中不再需要的孤立软件包（通常是依赖项）。
- **示例**:
  ```bash
  sudo apt autoremove  # 自动删除系统中不再需要的软件包
  ```

### `yum` (适用于基于Red Hat的系统，例如CentOS)

#### `yum update`
- **描述**: 更新系统中的所有包和依赖。
- **示例**:
  ```bash
  sudo yum update  # 更新所有包和系统中的依赖关系
  ```

#### `yum install`
- **描述**: 安装新的软件包。
- **示例**:
  ```bash
  sudo yum install package_name  # 安装名为package_name的软件包
  ```

#### `yum remove`
- **描述**: 删除已安装的软件包。
- **示例**:
  ```bash
  sudo yum remove package_name  # 删除名为package_name的软件包
  ```

#### `yum search`
- **描述**: 搜索可用的软件包。
- **示例**:
  ```bash
  yum search package_name  # 搜索与package_name相关的可用软件包
  ```

### `dnf` (适用于较新版本的基于Red Hat的系统，例如Fedora)

#### `dnf update`
- **描述**: 更新系统中的所有包和依赖。
- **示例**:
  ```bash
  sudo dnf update  # 更新所有包和系统中的依赖关系
  ```

#### `dnf install`
- **描述**: 安装新的软件包。
- **示例**:
  ```bash
  sudo dnf install package_name  # 安装名为package_name的软件包
  ```

#### `dnf remove`
- **描述**: 删除已安装的软件包。
- **示例**:
  ```bash
  sudo dnf remove package_name  # 删除名为package_name的软件包
  ```

#### `dnf search`
- **描述**: 搜索可用的软件包。
- **示例**:
  ```bash
  dnf search package_name  # 搜索与package_name相关的可用软件包
  ```

### `pacman` (适用于Arch Linux)

#### `pacman -Syu`
- **描述**: 同步包数据库并更新系统。
- **示例**:
  ```bash
  sudo pacman -Syu  # 更新系统并同步包数据库
  ```

#### `pacman -S`
- **描述**: 安装新的软件包。
- **示例**:
  ```bash
  sudo pacman -S package_name  # 安装名为package_name的软件包
  ```

#### `pacman -R`
- **描述**: 删除已安装的软件包。
- **示例**:
  ```bash
  sudo pacman -R package_name  # 删除名为package_name的软件包
  ```

#### `pacman -Ss`
- **描述**: 搜索可用的软件包。
- **示例**:
  ```bash
  pacman -Ss package_name  # 搜索与package_name相关的可用软件包
  ```


## 6. **压缩与解压**
```bash
`tar`: 用于创建、解压或查看压缩包（如`tar -czvf`创建一个压缩文件，`tar -xzvf`解压文件）。
`gzip` / `gunzip`: 压缩和解压文件。
`zip` / `unzip`: 用于压缩和解压zip格式的文件。
```
#### `tar`
- **描述**: 打包和解包文件。
- **示例**:
  ```bash
  tar -cvf archive.tar directory/      # 将目录打包为archive.tar
  tar -xvf archive.tar                 # 解包archive.tar
  tar -zcvf archive.tar.gz directory/  # 将目录打包并压缩为gzip格式的archive.tar.gz
  tar -zxvf archive.tar.gz             # 解压并解包archive.tar.gz
  ```

#### `zip` / `unzip`
- **描述**: 压缩和解压缩ZIP文件。
- **示例**:
  ```bash
  zip archive.zip file1 file2  # 将file1和file2压缩为archive.zip
  unzip archive.zip            # 解压缩archive.zip
  ```



## 7. **系统信息**
```bash
`uname`: 显示系统信息。
`hostname`: 显示或设置系统的主机名。
`whoami`: 显示当前用户的用户名。
`id`: 显示当前用户的ID信息。
`dmesg`: 显示系统引导日志信息。
`lsb_release`: 显示Linux发行版信息。
```
### `uname`
- **描述**: 显示系统的基本信息，如操作系统名称、内核版本等。
- **示例**:
  ```bash
  uname           # 显示操作系统的名称
  uname -a        # 显示系统的所有信息，包括内核版本、主机名、处理器架构等
  uname -r        # 显示正在运行的内核版本
  ```

### `hostname`
- **描述**: 显示或设置系统的主机名。
- **示例**:
  ```bash
  hostname        # 显示当前系统的主机名
  sudo hostname new_hostname  # 设置新的主机名（需要管理员权限）
  ```

### `whoami`
- **描述**: 显示当前登录用户的用户名。
- **示例**:
  ```bash
  whoami          # 显示当前用户的用户名
  ```

### `id`
- **描述**: 显示当前用户的用户ID (UID)、组ID (GID) 及所属组的信息。
- **示例**:
  ```bash
  id              # 显示当前用户的UID、GID和所属组信息
  id username     # 显示指定用户的UID、GID和所属组信息
  ```

### `dmesg`
- **描述**: 显示内核启动时产生的消息（也称为启动日志），通常用于诊断启动问题或硬件故障。
- **示例**:
  ```bash
  dmesg           # 显示系统启动以来的内核消息
  dmesg | grep error  # 仅显示与错误相关的消息
  ```

### `lsb_release`
- **描述**: 显示Linux发行版的相关信息，例如发行版名称、版本号、代号等。
- **示例**:
  ```bash
  lsb_release -a  # 显示所有发行版相关的信息
  lsb_release -d  # 仅显示发行版描述信息
  ```

### `uptime`
- **描述**: 显示系统的启动时间、运行时间、当前用户数、平均负载等信息。
- **示例**:
  ```bash
  uptime          # 显示系统的启动时间和平均负载
  ```

### `df`
- **描述**: 显示文件系统的磁盘使用情况。
- **示例**:
  ```bash
  df              # 显示所有挂载的文件系统的磁盘使用情况
  df -h           # 以人类可读的格式（如GB、MB）显示磁盘使用情况
  ```

### `free`
- **描述**: 显示系统的内存使用情况，包括物理内存和交换内存。
- **示例**:
  ```bash
  free            # 显示内存的使用情况
  free -h         # 以人类可读的格式显示内存使用情况
  ```

### `top`
- **描述**: 实时显示系统的任务和进程信息，包括CPU和内存的使用情况。
- **示例**:
  ```bash
  top             # 实时显示系统任务管理器信息
  ```

### `ps`
- **描述**: 显示当前系统运行的进程。
- **示例**:
  ```bash
  ps              # 显示当前用户在终端下运行的进程
  ps aux          # 显示系统中所有运行的进程
  ```

这些命令可以帮助你获取关于Linux系统的各种信息，从基本的系统信息到详细的内存和进程状态。
以下是一些常用的进程管理命令的详细描述和示例：

## 8. **进程管理**
```bash
`bg`: 将进程放到后台运行。
`fg`: 将后台进程带到前台运行。
`jobs`: 显示当前用户的作业列表。
```
### `bg`
- **描述**: 将一个挂起的进程放到后台继续运行。通常用于将暂停的前台作业转为后台作业。
- **示例**:
  ```bash
  bg             # 将最近暂停的作业放到后台运行
  bg %1          # 将作业ID为1的作业放到后台运行
  ```

### `fg`
- **描述**: 将后台的某个作业带到前台运行。可以用来恢复在后台运行的作业到前台继续执行。
- **示例**:
  ```bash
  fg             # 将最近的后台作业带到前台运行
  fg %1          # 将作业ID为1的作业带到前台运行
  ```

### `jobs`
- **描述**: 显示当前用户的作业列表，列出所有在当前shell会话中启动的作业，包括后台运行和暂停的作业。
- **示例**:
  ```bash
  jobs           # 列出当前用户的所有作业及其状态
  ```

### `kill`
- **描述**: 向指定进程发送信号。默认信号是SIGTERM (15)，用来终止进程。
- **示例**:
  ```bash
  kill 1234      # 终止进程ID为1234的进程
  kill -9 1234   # 强制终止进程ID为1234的进程（发送SIGKILL信号）
  ```

### `killall`
- **描述**: 向指定名称的所有进程发送信号。常用于终止所有匹配指定名称的进程。
- **示例**:
  ```bash
  killall firefox  # 终止所有名称为"firefox"的进程
  ```

### `pkill`
- **描述**: 根据进程名称或其他属性（如用户、会话）发送信号，与`killall`类似但更灵活。
- **示例**:
  ```bash
  pkill firefox  # 终止名称为"firefox"的所有进程
  pkill -u user1 # 终止用户"user1"的所有进程
  ```

### `ps`
- **描述**: 列出当前系统中的进程信息。
- **示例**:
  ```bash
  ps              # 显示当前用户在终端下运行的进程
  ps aux          # 显示系统中所有运行的进程，包括其他用户的
  ps -ef          # 显示所有进程的完整格式信息
  ```

### `top`
- **描述**: 实时显示系统中所有进程的资源使用情况（CPU、内存等），按资源占用排序，方便监控和管理进程。
- **示例**:
  ```bash
  top             # 实时显示系统任务管理器信息
  ```

### `htop`
- **描述**: 类似于`top`，但有更友好的界面，支持交互操作，如直接终止进程。
- **示例**:
  ```bash
  htop            # 启动htop工具查看进程状态（需要安装）
  ```

### `nice`
- **描述**: 启动一个进程并设置其优先级（影响其CPU使用比例），较低的优先级意味着进程会更少地占用CPU时间。
- **示例**:
  ```bash
  nice -n 10 ./script.sh  # 以优先级10启动script.sh脚本（默认是0）
  ```

### `renice`
- **描述**: 修改正在运行的进程的优先级。
- **示例**:
  ```bash
  renice 5 -p 1234         # 将进程ID为1234的进程优先级设置为5
  ```

### `nohup`
- **描述**: 让命令在退出终端后继续运行，通常用于长时间运行的任务。输出默认保存到`nohup.out`文件中。
- **示例**:
  ```bash
  nohup ./long_task.sh &   # 让long_task.sh脚本在后台运行，即使终端关闭也不会中断
  ```

### `strace`
- **描述**: 跟踪系统调用和信号，用于调试和分析进程的行为。
- **示例**:
  ```bash
  strace -p 1234           # 跟踪进程ID为1234的进程的系统调用
  ```



## 9. **常用的 VI 编辑器操作命令**

| VI命令              | 功能                                           | 区域        |
|-------------------------|------------------------------------------------|-------------|
| vi filename             | 打开filename文件                               | 文本编辑    |
| gvim filename           | 在另一个窗口打开filename文件                   ||
| :w                      | 保存文件                                       ||
| :w vpset.net            | 保存至vpset.net 文件                           ||
| :q                      | 退出编辑器                                     ||
| :q!                     | 退出编辑器，且不保存                           ||
| :wq                     | 退出编辑器，且保存文件                         ||
| a                       | 在当前光标位置的右边添加文本                   | 插入文本    |
| i                       | 在当前光标位置的左边添加文本                   ||
| A                       | 在当前行的末尾位置添加文本                     ||
| J                       | 合并光标所在行及下一行为一行                   ||
| I                       | 在当前行的开始处添加文本                       ||
| O                       | 在当前行的上面新建一行                         ||
| o                       | 在当前行的下面新建一行                         ||
| r                       | 替换光标所在处的字符                           | 替换        |
| R                       | 替换光标所到之处的字符，直到按下ESC键为止     |             |
| :s/old/new             | 用new替换行中首次出现的old ||
| :s/old/new/g           | 用new替换行中所有的old     ||
| :n,m s/old/new/g        | 用new替换从n到m行中所有的old                  ||
| :%s/old/new/g           | 用new替换当前文件里所有的old                  ||
| h                       | 向左                                           | 移动光标    |
| j                       | 向下                                           ||
| k                       | 向上                                           ||
| l                       | 向右                                           ||
| 空格键                   | 向右                                           ||
| Backspace               | 向左                                           |             |
| Enter                   | 移动到下一行首                                 |             |
| 横线-                   | 移动到上一行首                                 |             |
| ctrl+b                  | 屏幕往"后"移动一页                             |             |
| ctrl+f                  | 屏幕往"前"移动一页                             |             |
| ctrl+u                  | 屏幕往"后"移动半页                             |             |
| ctrl+d                  | 屏幕往"前"移动半页                             |             |
| 数字0                   | 移到文章的开头（暂不可用）                    |             |
| $                       | 移动到光标所在行的"行尾"                      |             |
| ^                       | 移动到光标所在行的"行首"                      |             |
| w                       | 光标跳到下个字的开头                           |             |
| e                       | 光标跳到下个字的字尾                           |             |
| b                       | 光标回到上个字的开头                           |             |
| #l                      | 光标移到该行的第#个位置，如：5l,56l           |             |
| x                       | 每按一次，删除光标所在位置的"后面"一个字符   | 删除文本    |
| #x                      | 例如，「6x」表示删除光标所在位置的"后面"6个字符 |             |
| X                       | 大写的X，每按一次，删除光标所在位置的"前面"一个字符 |             |
| #X                      | 例如，「20X」表示删除光标所在位置的"前面"20个字符 |             |
| dd                      | 删除光标所在行                                 |             |
| #dd                     | 从光标所在行开始删除#行                        |             |
| u                       | 撤销上一步操作                                 | 恢复        |
| U                       | 撤销对当前行的所有操作                         |             |
| yw                      | 将光标所在之处到字尾的字符复制到缓冲区中       | 复制粘贴    |
| #yw                     | 复制#个字到缓冲区                              |             |
| yy                      | 复制光标所在行到缓冲区                         |             |
| #yy                     | 例如，「6yy」表示拷贝从光标所在的该行"往下数"6行文字 |             |
| y^                      | 复制从光标到行首的内容                         |             |
| y$                      | 复制从光标到行尾的内容                         |             |
| p                       | 将缓冲区内的字符贴到光标所在位置后             |             |
| P                       | 将缓冲区内的字符贴到光标所在位置前             |             |
| n+                      | 向下跳转n行                                    | 跳到指定行  |
| n-                      | 向上跳转n行                                    |             |
| nG                      | 跳到行号为n的行                                |             |
| G                       | 调至文件的底部                                 |             |
| ctrl+g                  | 列出光标所在行的行号                           |             |
| /vpser                  | 向光标下搜索vpser字符串                        | 搜索        |
| ？vpser                 | 向光标上搜索vpser字符串                        |             |
| n                       | 向下搜索前一个搜索动作                         |             |
| N                       | 向上搜索前一个搜索动作                         |             |
| :set nu                 | 显示行号                                       | 设置行号    |
| :set nonu               | 取消显示行号                                   |             |
