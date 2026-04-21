# 附录 B：命令速查补全表

这个附录保留旧版 `LinuxCMD_Basics.md` 中较细的命令点，作为正文教程的补充索引。

## 文件和目录

```bash
ls
ls -l
ls -a
ls -lh
cd /path
cd ..
cd ~
pwd
mkdir new_folder
mkdir -p dir1/dir2/dir3
rmdir empty_folder
rm file.txt
rm -r directory
rm -rf directory
cp file1.txt file2.txt
cp -r directory1 directory2
mv old_name.txt new_name.txt
mv file.txt /path/to/directory
touch newfile.txt
find /path -name "*.txt"
find /path -type d -name "backup"
```

提醒：`rm -rf` 会强制递归删除，执行前确认路径。

## 文件查看和编辑

```bash
cat file.txt
cat file1.txt file2.txt
more file.txt
less file.txt
head file.txt
head -n 20 file.txt
tail file.txt
tail -n 20 file.txt
tail -f log.txt
nano file.txt
vi file.txt
vim file.txt
grep "keyword" file.txt
grep -r "keyword" /path
awk '{print $1}' file.txt
sed 's/old/new/g' file.txt
```

## Vim 常用操作

```text
i       进入插入模式
a       在光标后进入插入模式
o       在下一行新建并进入插入模式
Esc     回到普通模式
:w      保存
:q      退出
:wq     保存并退出
:q!     不保存强制退出
dd      删除当前行
yy      复制当前行
p       粘贴
/word   搜索 word
n       跳到下一个搜索结果
u       撤销
```

## 权限和用户

```bash
chmod +x script.sh
chmod 644 file.txt
chmod 755 script.sh
chown user file.txt
chown user:group file.txt
chown -R user:group folder/
sudo command
```

提醒：谨慎使用 `chmod 777` 和 `chown -R`。

## 系统信息

```bash
uname -a
hostname
whoami
id
dmesg
dmesg | tail
lsb_release -a
uptime
df -h
du -sh *
free -h
top
ps aux
```

## 进程管理

```bash
ps aux
ps aux | grep name
top
htop
kill PID
killall name
pkill name
command &
jobs
fg
bg
nice -n 10 command
renice 10 -p PID
nohup command &
strace command
```

- `nice`: 启动命令时设置优先级。
- `renice`: 调整已有进程的优先级。
- `nohup`: 终端关闭后继续运行命令。
- `strace`: 跟踪系统调用，常用于排查程序卡住或文件访问问题。

## 网络操作

```bash
ping example.com
ifconfig
ip addr
ssh user@server_ip
ssh -p 2222 user@server_ip
scp file.txt user@server_ip:/path
scp -P 2222 file.txt user@server_ip:/path
wget https://example.com/file.tar.gz
curl https://example.com
curl -I https://example.com
```

## 软件包管理

### apt

```bash
sudo apt update
sudo apt upgrade
sudo apt install package-name
sudo apt remove package-name
sudo apt autoremove
```

### yum

```bash
sudo yum update
sudo yum install package-name
sudo yum remove package-name
sudo yum search keyword
```

### dnf

```bash
sudo dnf update
sudo dnf install package-name
sudo dnf remove package-name
sudo dnf search keyword
```

### pacman

```bash
sudo pacman -Syu
sudo pacman -S package-name
sudo pacman -R package-name
sudo pacman -Ss keyword
```

## 压缩与解压

```bash
tar -czvf archive.tar.gz folder/
tar -xzvf archive.tar.gz
zip -r archive.zip folder/
unzip archive.zip
```

