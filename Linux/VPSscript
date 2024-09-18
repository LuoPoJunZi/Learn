# 常用VPS脚本
部分脚本附带原脚本发布地址，可访问了解详细脚本细节。
## 1、DD重装脚本
### 史上最强脚本
```markdown
wget --no-check-certificate -qO InstallNET.sh 'https://raw.githubusercontent.com/leitbogioro/Tools/master/Linux_reinstall/InstallNET.sh' && chmod a+x InstallNET.sh && bash InstallNET.sh -debian 12 -pwd 'password'
```
### 萌咖大佬的脚本
```markdown
 bash <(wget --no-check-certificate -qO- 'https://raw.githubusercontent.com/MoeClub/Note/master/InstallNET.sh') -d 11 -v 64 -p 密码 -port 端口 -a -firmware
```
### DD windows（使用史上最强DD脚本）
支持Windows 10、Windows 11、Windows Server 2012 R2、Windows Server 2016、Windows Server 2019、Windows Server 2022
```markdown
wget --no-check-certificate -qO InstallNET.sh 'https://raw.githubusercontent.com/leitbogioro/Tools/master/Linux_reinstall/InstallNET.sh' && chmod a+x InstallNET.sh && bash InstallNET.sh -windows 10 -lang "cn"
```
```markdown
账户：Administrator
密码：Teddysun.com
```
## 2、综合测试脚本
### 融合怪
```markdown
bash <(wget -qO- --no-check-certificate https://gitlab.com/spiritysdx/za/-/raw/main/ecs.sh)
```
### NodeBench
```markdown
bash <(curl -sL https://raw.githubusercontent.com/LloydAsp/NodeBench/main/NodeBench.sh)
```
### yabs
```markdown
curl -sL yabs.sh | bash
```
### 使用gb5测试yabs
```markdown
curl -sL yabs.sh | bash -5
```
## 3、性能测试
### gb5专测脚本
```markdown
bash <(curl -sL bash.icu/gb5)
```
## 4、流媒体及IP质量测试
### 最常用版本
```markdown
bash <(curl -L -s media.ispvps.com)
```
### 原生检测脚本
```markdown
bash <(curl -L -s check.unlock.media)
```
### 准确度最高
```markdown
bash <(curl -L -s https://github.com/1-stream/RegionRestrictionCheck/raw/main/check.sh)
```
### IP质量体检脚本
```markdown
bash <(curl -sL IP.Check.Place)
```
### Chatpt APP 解锁检测
- 安卓
```markdown
curl android.chat.openai.com
```
- 苹果
```markdown
curl ios.chat.openai.com
```
- 结果解析：https://www.nodeseek.com/post-31717-1
## 5、测速脚本
### 三网测速脚本
- Speedtest
```markdown
bash <(curl -sL bash.icu/speedtest)
```
- Taier
```markdown
bash <(curl -sL res.yserver.ink/taier.sh)
```
- hyperspeed
```markdown
bash <(curl -Lso- https://bench.im/hyperspeed)
```
- 全球测速
```markdown
curl -sL network-speed.xyz | bash
```
## 6、回程测试
### 直接显示回程（小白用这个）
```markdown
curl https://raw.githubusercontent.com/ludashi2020/backtrace/main/install.sh -sSf | sh
```
### 回程详细测试（推荐）
```markdown
wget https://ghproxy.com/https://raw.githubusercontent.com/vpsxb/testrace/main/testrace.sh -O testrace.sh && bash testrace.sh
```
## 7、功能脚本
### Fail2ban
```markdown
wget --no-check-certificate https://raw.githubusercontent.com/FunctionClub/Fail2ban/master/fail2ban.sh && bash fail2ban.sh 2>&1 | tee fail2ban.log
```
### 开启BBR
- 一键开启BBR，适用于较新的Debian、Ubuntu
```markdown
echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
sysctl -p
sysctl net.ipv4.tcp_available_congestion_control
lsmod | grep bbr
```
- 多功能BBR安装脚本
```markdown
wget -N --no-check-certificate "https://gist.github.com/zeruns/a0ec603f20d1b86de6a774a8ba27588f/raw/4f9957ae23f5efb2bb7c57a198ae2cffebfb1c56/tcp.sh" && chmod +x tcp.sh && ./tcp.sh
```
### TCP窗口调优
```markdown
wget http://sh.nekoneko.cloud/tools.sh -O tools.sh && bash tools.sh
```
### 测试访问优先级
```markdown
curl ip.sb
```
### 添加SWAP
```markdown
wget https://www.moerats.com/usr/shell/swap.sh && bash swap.sh
```
### 25端口开放测试
```markdown
telnet smtp.aol.com 25
```
## 8、一键安装常用环境及软件
### docker
- 国外专用
```markdown
curl -sSL https://get.docker.com/ | sh
```
- 国内专用
```markdown
curl -sSL https://get.daocloud.io/docker | sh
```
### Python
```markdown
curl -O https://raw.githubusercontent.com/lx969788249/lxspacepy/master/pyinstall.sh && chmod +x pyinstall.sh && ./pyinstall.sh
```
### WARP
```markdown
wget -N https://gitlab.com/fscarmen/warp/-/raw/main/menu.sh && bash menu.sh
```
### Aria2一键安装脚本
```markdown
wget -N git.io/aria2.sh && chmod +x aria2.sh && ./aria2.sh
```
### aaPanel(宝塔国际版)
```markdown
URL=https://www.aapanel.com/script/install_7.0_en.sh && if [ -f /usr/bin/curl ];then curl -ksSO "$URL" ;else wget --no-check-certificate -O install_7.0_en.sh "$URL";fi;bash install_7.0_en.sh aapanel
```
### 宝塔
```markdown
url=https://download.bt.cn/install/install_lts.sh;if [ -f /usr/bin/curl ];then curl -sSO $url;else wget -O install_lts.sh $url;fi;bash install_lts.sh ed8484bec
```
### 宝塔开心版
- 访问：https://bt.sb/bbs/forum-37-1.html
## 9、综合功能脚本
### 科技lion
```markdown
curl -sS -O https://kejilion.pro/kejilion.sh && chmod +x kejilion.sh && ./kejilion.sh
```
### SKY-BOX
```markdown
wget -O box.sh https://raw.githubusercontent.com/BlueSkyXN/SKY-BOX/main/box.sh && chmod +x box.sh && clear && ./box.sh
```

## 论坛博客[原帖](https://www.nodeseek.com/post-143131-1)

## 记得关注[我](https://www.nodeseek.com/space/18714#/general)！！！
