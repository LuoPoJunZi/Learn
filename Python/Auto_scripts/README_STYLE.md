# Auto_scripts README 写作规范

这个规范用于统一 [Auto_scripts](README.md) 下每个脚本目录的说明文档。后续新增或整理脚本时，建议尽量按这个结构写。

## 推荐结构

````markdown
# 脚本名称

一句话说明这个脚本解决什么问题。

## 功能

- 功能一
- 功能二

## 适合场景

- 场景一
- 场景二

## 运行环境

- Python 版本
- 第三方依赖

## 安装依赖

```bash
pip install package-name
```

## 使用方式

```bash
python script_name.py
```

## 输入和输出

| 项目 | 说明 |
| :--- | :--- |
| 输入 | 输入文件、目录、网址或参数 |
| 输出 | 输出文件、目录或终端结果 |

## 修改配置

说明脚本里哪些变量适合新手修改。

## 常见问题

### 问题一

回答。

## 风险提醒

- 是否会删除、覆盖、移动文件
- 是否会请求网络
- 是否需要账号、密码、Token
````

## 各类脚本的重点

### 文件处理脚本

例如：

- `sort_files`
- `rename_files`
- `remove_empty_folders`
- `find_replace`

必须写清楚：

- 会处理哪个目录。
- 是否会移动、重命名、删除文件。
- 是否会覆盖已有文件。
- 是否建议先在测试目录运行。

### 网络请求脚本

例如：

- `check_website_status`
- `download_images`
- `scrape_data`

必须写清楚：

- 请求的目标是什么。
- 是否需要控制请求频率。
- 是否需要遵守目标网站规则。
- 网络失败时脚本会怎么处理。

### 数据文件脚本

例如：

- `read_excel`
- `remove_duplicates`
- `extract_text_from_pdf`
- `execute_query`

必须写清楚：

- 支持什么文件格式。
- 输入文件路径在哪里改。
- 输出文件会保存到哪里。
- 是否依赖 `pandas`、`openpyxl`、`PyPDF2` 等库。

### 账号和 API 脚本

例如：

- `post_tweet`
- `send_personalized_email`

必须写清楚：

- 需要哪些凭证。
- 凭证不要写入代码。
- 推荐使用环境变量。
- 发送前先用测试账号或少量收件人验证。

## 新手友好写法

推荐写：

````markdown
先把 `folder_path` 改成你自己的测试目录，例如：

```python
folder_path = r"D:\test-files"
```
````

不推荐只写：

```markdown
修改路径后运行。
```

原因是新手往往不知道“路径”具体是哪一行，也不知道 Windows 路径前面的 `r` 有什么用。

## 提交前检查

整理脚本 README 后，建议检查：

- 标题是否清楚。
- 是否能一眼看出脚本用途。
- 是否有运行命令。
- 是否列出依赖。
- 是否说明输入输出。
- 是否提醒危险操作。
- 是否有敏感信息提醒。
- 链接是否能打开。
