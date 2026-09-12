# scripts

这个目录保存仓库维护脚本，不需要安装 npm 或额外的 Python 检查库。

## 文件

- [check-all.ps1](check-all.ps1)：统一运行仓库结构、代码、文档和 Git 空白检查
- [check-repository.py](check-repository.py)：检查必要根文件、一级 README、跨平台路径冲突，以及误提交的缓存、运行结果和源码压缩包
- [check-code.ps1](check-code.ps1)：统一运行 Python、JavaScript、PowerShell、HTML 和 Matlab 文件名检查
- [check-python-code.py](check-python-code.py)：使用 Python 标准库检查语法、行长、危险导入、网络超时、文本编码和旧 API
- [check-html-code.py](check-html-code.py)：检查 HTML 重复 ID、图片替代文本和按钮类型
- [check-docs.ps1](check-docs.ps1)：检查 Markdown 本地链接、代码块闭合情况，以及一级目录是否包含 README
- [MAINTAINER_GUIDE.md](MAINTAINER_GUIDE.md)：文档维护者指南，说明新增教程、更新导航和排查检查失败的流程

日常提交前只需要运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-all.ps1
```

总入口会依次检查：

1. 仓库必要文件、一级目录入口、跨平台路径和不应提交的生成文件。
2. Python、JavaScript、PowerShell、HTML 和 Matlab 示例。
3. Markdown 本地链接与代码块。
4. 已暂存和未暂存差异中的空白错误。

`check-all.ps1` 需要系统能够找到 `git` 和 `python`；仓库存在 JavaScript 文件时还需要 `node`。它不会运行网络、删除、邮件或算法示例，只做静态检查。需要定位单项问题时，仍可单独运行 `check-code.ps1` 或 `check-docs.ps1`。

GitHub Actions 的 [quality 工作流](../.github/workflows/quality.yml) 使用同一个总入口，避免本地规则和 CI 规则不一致。
