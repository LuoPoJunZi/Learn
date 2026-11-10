# 自动化脚本统一运行与安全说明

这个目录中的脚本最初以“逐行解释一个功能”为主，因此部分 README 保留了早期基础版本的完整代码和扩展思路。当前 `.py` 文件已经做过安全与兼容性标准化；学习概念时可以继续阅读原说明，实际运行时应以同目录中的 `.py` 文件为准。

## 适合人群

- 准备第一次运行自动化脚本的新手。
- 想知道本轮代码标准化改了什么的人。
- 需要批量处理文件、访问网络或使用账号凭证的人。

## 统一规则

- 网络请求设置超时，避免脚本无限等待。
- 文本文件明确使用 UTF-8，减少不同系统上的乱码。
- API Key、Token 和邮箱密码从环境变量读取，不写在源码里。
- 删除、重命名、替换和移动文件的脚本默认只预演。
- 可能覆盖原文件的处理默认生成新文件，或在覆盖前主动检查。
- 第三方库使用当前 API，例如 `PdfReader`、`extract_text()` 和 `Image.Resampling.LANCZOS`。

## 安全行为速查

| 脚本 | 当前保护 | 真正执行前要做什么 |
| :--- | :--- | :--- |
| `find_replace` | `dry_run=True`，默认只统计替换数量 | 确认备份后传入 `dry_run=False` |
| `rename_files` | `dry_run=True`，默认只显示改名计划 | 检查重名冲突后传入 `dry_run=False` |
| `remove_empty_folders` | `dry_run=True`，默认只列出空目录 | 确认目录范围后传入 `dry_run=False` |
| `sort_files` | `dry_run=True`，默认只显示移动计划 | 检查目标目录后传入 `dry_run=False` |
| `remove_duplicates` | 默认保存为 `*_deduplicated.xlsx` | 先核对新文件，再决定是否替换原文件 |
| `Grabpicture` | 要求目标目录为空 | 使用新建测试目录，并遵守目标网站规则 |
| `download_images` | 设置请求超时并记录失败项 | 使用测试接口和独立输出目录 |
| `post_tweet` | 从环境变量读取 X/Twitter 凭证 | 先用测试账号验证权限和发送额度 |
| `send_personalized_email` | 从环境变量读取 SMTP 凭证并使用 TLS | 先只填写自己的测试邮箱 |

## 预演与执行示例

以文本替换为例，先进入脚本目录并打开 Python：

```powershell
cd Python\Auto_scripts\find_replace
python
```

先预演：

```python
from find_replace import find_replace

find_replace("example.txt", "旧文本", "新文本")
```

确认输出数量和备份都正确后，再执行：

```python
find_replace("example.txt", "旧文本", "新文本", dry_run=False)
```

`rename_files`、`remove_empty_folders` 和 `sort_files` 使用同样的 `dry_run=False` 方式切换到实际执行。不要把第一次测试直接对准下载目录、照片目录或项目根目录。

## 路径、链接与并发边界

- 重命名替换文本不得含路径分隔符；源符号链接不处理，目标悬空链接也算冲突。
- 分类脚本不跟随符号链接目标，也不会把普通文件当作分类目录。
- 空目录预演包含“删掉子目录后变空”的父目录，真实执行仍重新检查目录状态。
- 这些保护针对本地学习和正常使用，不提供对抗并发替换路径的安全沙箱；在无其他写入者的测试副本里执行，保留备份。
- 回归测试仅在自动创建的临时目录里执行移动/重命名/清理，不接触用户真实数据。

## 环境变量示例

PowerShell 临时设置 X/Twitter 凭证：

```powershell
$env:X_API_KEY="你的 API Key"
$env:X_API_SECRET="你的 API Secret"
$env:X_ACCESS_TOKEN="你的 Access Token"
$env:X_ACCESS_TOKEN_SECRET="你的 Access Token Secret"
```

设置 SMTP 凭证：

```powershell
$env:SMTP_SENDER="you@example.com"
$env:SMTP_PASSWORD="应用专用密码"
```

这些变量只对当前终端会话生效。不要把真实值写进 README、截图、`.py` 文件或 Git 提交。

## 代码与文档如何对应

- 子目录 README 中的长代码块保留原始教学脉络，适合理解函数、循环和模块用途。
- 同目录 `.py` 文件是当前可运行版本，包含本页列出的安全修正。
- 如果说明文字与脚本细节不同，以脚本和本页为准，并同步提交文档修正。

## 下一步入口

- [自动化脚本索引](README.md)
- [依赖总览](DEPENDENCIES.md)
- [代码规范](../../docs/CODE_STYLE.md)
- [Python 环境与依赖管理](../ENVIRONMENT_GUIDE.md)

