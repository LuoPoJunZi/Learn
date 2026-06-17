# Learn

记录学习过程、整理常用教程、收藏一些实用脚本和示例代码。

这个仓库里的内容一部分来自开源项目和公开资料，一部分是我结合 AI 整理后的学习笔记。你可以把它当成一个个人知识库，也可以按目录直接查找需要的内容。

## 推荐入口

- [GitHub 新手入门教程](Github/README.md)
- [Linux 新手入门与 VPS 实用手册](Linux/README.md)
- [Markdown 用法教程](Markdown/README.md)
- [Python 学习与自动化脚本](Python/README.md)
- [Matlab 学习与算法资料库](Matlab/README.md)

## 目录导航

### GitHub

[Github](Github/README.md) 是一套面向新手的 GitHub 教程，包含：

- Git 与 GitHub 的区别
- Git 安装、账号配置、SSH 密钥
- 工作区、暂存区、本地仓库、远程仓库
- 常用 Git 命令
- 本地项目上传到 GitHub
- VS Code 图形化 Git 工作流
- 分支、Pull Request、冲突解决
- `.gitignore`、Commit 规范、Issue 写法
- GitHub 脚本短链和 Cloudflare Workers 实用技巧

### Linux

[Linux](Linux/README.md) 已整理成“基础教程 + 系统管理 + SSH 安全 + VPS 工具”的结构，包含：

- Linux 入门概念
- 文件和目录操作
- 文件查看、编辑、搜索
- 权限与所有者
- 进程、服务、系统资源
- 软件包管理
- 网络工具与 SSH 登录
- SSH 安全加固和日志排查
- VPS 常用脚本速查表
- VPS 操作安全清单
- 旧内容迁移索引和命令速查附录

### Markdown

[Markdown](Markdown/README.md) 是 Markdown 语法教程，包含：

- 标题、段落、换行
- 加粗、斜体、删除线
- 引用、列表、任务列表
- 链接、图片
- 行内代码、代码块
- 表格、分割线、脚注
- 数学公式
- Mermaid 图表
- CSDN、Hexo/安知鱼主题扩展语法说明

### HTML

[HTML](HTML/) 目录保存一些网页示例：

- [Birthday](HTML/Birthday/README.md)：生日快乐网页
- [Love](HTML/Love/README.md)：表白网页示例，部分代码来自开源项目并经过整理注释

更多表白网页示例可参考：[Awesome-Love-Code](https://github.com/sun0225SUN/Awesome-Love-Code)

### Mathematica

[Mathematica](Mathematica/README.md) 记录 Mathematica 基础命令和入门笔记。

### Matlab

[Matlab](Matlab/README.md) 目录包含 Matlab 学习资料和算法代码：

- `Basics`：Matlab 基础命令、并行计算等笔记
- `Multi-Objective Optimization`：多目标优化算法示例
- `Neural Network`：神经网络相关内容

其中 Matlab 基础内容参考了郭彦甫老师的 MATLAB 课程，并结合 AI 整理。

### Python

[Python](Python/README.md) 目录包含 Python 脚本和示例项目：

- [Basics](Python/Basics/README.md)：Python 基础教程，适合新手入门
- [Auto_scripts](Python/Auto_scripts/README.md)：日常自动化脚本，例如磁盘检查、网站状态检查、批量重命名、PDF 文本提取等
- `Grab`：图片抓取相关脚本
- `Love`：Python 表白小程序示例

### 文档维护

- [文档整理规范](docs/STYLE_GUIDE.md)
- [文档检查脚本](scripts/check-docs.ps1)

## 使用建议

如果你是新手，建议先读：

1. [Markdown 用法教程](Markdown/README.md)：先学会写文档。
2. [GitHub 新手入门教程](Github/README.md)：学会管理和上传项目。
3. [Linux 新手入门与 VPS 实用手册](Linux/README.md)：再学习服务器和命令行。
4. [Python 基础教程](Python/Basics/README.md)：开始写自己的自动化脚本。

如果你只是想查命令，可以直接进入对应目录的 README 或 docs。

维护文档时，可以运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-docs.ps1
```

## 说明

本仓库内容主要用于学习和记录。涉及一键脚本、VPS 操作、远程安装命令时，请先确认来源、阅读脚本内容，并在重要环境中做好备份。
