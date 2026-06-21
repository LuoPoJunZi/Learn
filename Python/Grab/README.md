# Python 图片抓取示例

这个目录保存图片抓取相关脚本，目前包含：

- [Grabpicture.py](Grabpicture.py)：根据关键词搜索并下载图片的示例脚本

这类脚本适合用来学习：

- `requests` 如何发送网页请求
- `BeautifulSoup` 如何解析网页内容
- 正则表达式如何提取图片地址
- Python 如何创建目录、保存二进制文件

## 运行环境

建议使用 Python 3.10 或更新版本。

脚本依赖：

```bash
pip install requests beautifulsoup4
```

如果你使用虚拟环境，建议先回到 `Python` 目录或自己的测试目录，再创建环境：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install requests beautifulsoup4
```

macOS / Linux 激活方式通常是：

```bash
source .venv/bin/activate
```

## 如何运行

进入当前目录：

```powershell
cd Python\Grab
python Grabpicture.py
```

运行后根据终端提示输入关键词、下载数量、保存路径等信息。

如果脚本运行失败，优先检查：

- 是否安装了 `requests` 和 `beautifulsoup4`
- 当前网络是否可以访问目标站点
- 保存目录是否存在，是否有写入权限
- 关键词是否能搜索到图片

## 新手阅读顺序

建议按下面顺序理解脚本：

1. 先看文件开头的 `import`，了解脚本用到了哪些库。
2. 再看全局变量，理解下载数量、保存路径、图片列表放在哪里。
3. 阅读 `Find` 函数，理解如何查找图片地址。
4. 阅读 `recommend` 函数，理解如何解析相关推荐词。
5. 阅读下载函数，理解图片如何被保存到本地。
6. 最后看主流程，理解用户输入如何串起整个脚本。

## 修改建议

你可以从这些小改动开始练习：

- 修改默认保存目录。
- 限制最大下载数量，避免一次下载太多。
- 给文件名加时间戳，减少重名覆盖风险。
- 下载前打印保存路径，方便确认。
- 给失败的图片 URL 写入日志文件。

## 安全提醒

- 抓取图片前先确认目标网站规则，不要高频请求。
- 下载的图片可能有版权限制，不要直接用于商业用途。
- 不要把私人 Cookie、账号密码、Token 写进脚本。
- 批量下载前先用少量数量测试，例如 3 到 5 张。
- 如果网站结构变化，脚本里的正则表达式可能需要重新调整。

