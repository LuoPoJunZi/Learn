# Python 自动化脚本索引

这里整理一些日常可能用到的 Python 小工具。每个脚本目录通常包含一个 `.py` 文件和对应的 `README.md`。

## 快速选择

| 场景 | 脚本 | 说明 |
| :--- | :--- | :--- |
| 整理文件 | [sort_files](sort_files/) | 根据文件扩展名将目录中的文件组织到子目录中 |
| 删除空目录 | [remove_empty_folders](remove_empty_folders/) | 删除指定目录下的空文件夹 |
| 批量改名 | [rename_files](rename_files/) | 批量重命名目录中的文件 |
| 文本统计 | [count_words](count_words/) | 统计指定文件中的单词总数 |
| 文本替换 | [find_replace](find_replace/) | 在文件中查找并替换特定文本 |
| 网站检测 | [check_website_status](check_website_status/) | 检查指定网站是否可以正常访问 |
| 磁盘检测 | [check_disk_space](check_disk_space/) | 监控系统中的可用磁盘空间 |
| 图片下载 | [download_images](download_images/) | 从接口或网页批量下载图片 |
| 图片缩放 | [resize_image](resize_image/) | 调整图像大小 |
| 图片 OCR | [recognize_text](recognize_text/) | 从图像文件中提取文本 |
| PDF 文本 | [extract_text_from_pdf](extract_text_from_pdf/) | 从 PDF 文件中提取文本 |
| Excel 读写 | [read_excel](read_excel/) | 读取和写入 Excel 文件 |
| Excel 去重 | [remove_duplicates](remove_duplicates/) | 从 Excel 文件中移除重复行 |
| SQLite 查询 | [execute_query](execute_query/) | 连接 SQLite 数据库并执行 SQL 查询 |
| 邮件发送 | [send_personalized_email](send_personalized_email/) | 向多个收件人发送个性化电子邮件 |
| 社交发布 | [post_tweet](post_tweet/) | 使用 Tweepy 向 Twitter/X 发布推文 |
| 网页抓取 | [scrape_data](scrape_data/) | 从网站上抓取数据 |

## 按用途分类

### 文件处理

- [sort_files](sort_files/)
- [remove_empty_folders](remove_empty_folders/)
- [rename_files](rename_files/)
- [find_replace](find_replace/)
- [count_words](count_words/)

### 数据与文档

- [read_excel](read_excel/)
- [remove_duplicates](remove_duplicates/)
- [extract_text_from_pdf](extract_text_from_pdf/)
- [execute_query](execute_query/)

### 图片处理

- [download_images](download_images/)
- [resize_image](resize_image/)
- [recognize_text](recognize_text/)

### 网络与自动化

- [check_website_status](check_website_status/)
- [check_disk_space](check_disk_space/)
- [scrape_data](scrape_data/)

### 账号和 API

- [send_personalized_email](send_personalized_email/)
- [post_tweet](post_tweet/)

这类脚本通常需要账号、密码、Token 或 API Key。不要把真实凭证写进代码或提交到仓库。

## 运行方式

进入脚本目录后运行：

```bash
python script_name.py
```

如果脚本依赖第三方库，先安装依赖，例如：

```bash
pip install requests pandas openpyxl pillow tweepy
```

具体依赖以每个脚本 README 和 `.py` 文件中的 import 为准。

## 使用前检查

- 路径是否正确。
- 输入文件是否存在。
- 输出目录是否会覆盖已有文件。
- 网络请求是否符合目标网站规则。
- 邮件、Twitter/X、数据库等凭证是否通过安全方式传入。

