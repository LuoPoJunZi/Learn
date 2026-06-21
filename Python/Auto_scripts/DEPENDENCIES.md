# Python 自动化脚本依赖总览

这个文件汇总 [Auto_scripts](README.md) 中各脚本可能用到的依赖，方便新手在运行前先准备环境。

## 推荐环境

建议使用 Python 3.10 或更新版本。

为了避免依赖污染系统环境，建议使用虚拟环境：

```powershell
cd Python\Auto_scripts
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

macOS / Linux 激活方式通常是：

```bash
source .venv/bin/activate
```

## 依赖速查表

| 脚本目录 | 主要依赖 | 说明 |
| :--- | :--- | :--- |
| `check_disk_space` | 标准库 | 使用 `shutil` 检查磁盘空间 |
| `check_website_status` | `requests` | 检查网站是否可访问 |
| `count_words` | 标准库 | 文本读取和统计 |
| `download_images` | `requests` | 下载图片并保存到本地 |
| `execute_query` | 标准库 | 使用 `sqlite3` 查询 SQLite 数据库 |
| `extract_text_from_pdf` | `PyPDF2` | 提取 PDF 文本 |
| `find_replace` | 标准库 | 文件文本查找替换 |
| `post_tweet` | `tweepy` | 调用 Twitter/X API |
| `read_excel` | `pandas`、`openpyxl` | 读取和写入 Excel |
| `recognize_text` | `pytesseract`、`Pillow` | 图片 OCR 识别 |
| `remove_duplicates` | `pandas`、`openpyxl` | Excel 去重 |
| `remove_empty_folders` | 标准库 | 删除空文件夹 |
| `rename_files` | 标准库 | 批量重命名 |
| `resize_image` | `Pillow` | 调整图片尺寸 |
| `scrape_data` | `requests`、`beautifulsoup4` | 网页抓取和 HTML 解析 |
| `send_personalized_email` | 标准库 | 使用 `smtplib` 和 `email` 发送邮件 |
| `sort_files` | 标准库 | 文件分类移动 |

标准库表示 Python 自带，通常不需要额外安装。

## 一次性安装常用依赖

如果你想先把大部分脚本需要的第三方库装好，可以运行：

```bash
pip install requests beautifulsoup4 pandas openpyxl pillow PyPDF2 tweepy pytesseract
```

如果只运行某一个脚本，建议只安装它需要的库。例如只运行网页检测：

```bash
pip install requests
```

## OCR 额外说明

`recognize_text` 使用 `pytesseract`。它不仅需要 Python 包，还通常需要本机安装 Tesseract OCR 程序。

大致流程是：

1. 安装 Tesseract OCR。
2. 安装 Python 包：

```bash
pip install pytesseract pillow
```

3. 如果程序找不到 Tesseract，需要在脚本中配置可执行文件路径。

Windows 常见路径类似：

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

具体路径以你电脑实际安装位置为准。

## 账号和凭证

这些脚本可能涉及敏感信息：

- `post_tweet`
- `send_personalized_email`

不要把真实 API Key、Token、邮箱密码写进代码并提交到 GitHub。

更推荐使用环境变量：

```powershell
$env:API_KEY="你的密钥"
```

Python 中读取：

```python
import os

api_key = os.getenv("API_KEY")
```

## 运行前自查

运行脚本前，先确认：

- 是否进入了正确目录。
- 是否激活了虚拟环境。
- 是否安装了脚本需要的第三方库。
- 输入文件是否存在。
- 输出目录是否会覆盖已有文件。
- 网络、账号、权限是否准备好。
- 批量操作是否先用测试目录验证过。

