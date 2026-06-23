# scripts

这个目录保存仓库维护脚本。

## 文件

- [check-docs.ps1](check-docs.ps1)：检查 Markdown 本地链接、代码块闭合情况，以及一级目录是否包含 README
- [MAINTAINER_GUIDE.md](MAINTAINER_GUIDE.md)：文档维护者指南，说明新增教程、更新导航和排查检查失败的流程

运行方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-docs.ps1
```
