# GitHub 学习导读

这篇导读帮助你把 GitHub 目录里的零散知识串成一条路线。目标不是背命令，而是理解“修改从本地到远程仓库”的完整过程。

## 适合人群

- 只会下载代码，还不会提交代码的新手。
- 会一点 Git 命令，但不理解分支、PR、冲突的人。
- 想参与开源项目，或者加入团队协作的人。

## 学习目标

学完后，你应该能说清楚：

- Git 和 GitHub 分别解决什么问题。
- 工作区、暂存区、本地仓库、远程仓库之间如何流转。
- 为什么团队协作通常不直接改 `main`。
- Fork、Branch、Pull Request、Review 各自出现在哪一步。

## 推荐阅读顺序

1. [Git 与 GitHub 的区别](docs/part1-basics/1.1-what-is-git.md)
2. [账号注册与安装 Git](docs/part1-basics/1.2-setup.md)
3. [配置 SSH 密钥](docs/part1-basics/1.3-ssh-keys.md)
4. [工作区、暂存区、本地仓库](docs/part2-git-cmds/2.1-core-concepts.md)
5. [日常指令备忘录](docs/part2-git-cmds/2.2-daily-commands.md)
6. [用 VS Code 丝滑提交](docs/part2-git-cmds/2.4-vscode-workflow.md)
7. [分支](docs/part3-teamwork/3.1-branching.md)
8. [Pull Request](docs/part3-teamwork/3.2-pull-request.md)
9. [冲突解决](docs/part3-teamwork/3.3-merge-conflict.md)

## 多人协作流程图

```mermaid
flowchart LR
    A["同步 main"] --> B["创建功能分支"]
    B --> C["修改文件"]
    C --> D["提交 commit"]
    D --> E["推送分支"]
    E --> F["发起 Pull Request"]
    F --> G["Review 和讨论"]
    G --> H["修改并补充提交"]
    H --> I["合并到 main"]
```

## 常见概念误区

| 误区 | 正确理解 |
| :--- | :--- |
| GitHub 就是 Git | Git 是版本控制工具，GitHub 是远程托管和协作平台。 |
| `commit` 之后别人就能看到 | `commit` 只保存到本地，还要 `push` 到远程。 |
| 分支越多越乱 | 分支是隔离修改的工具，命名清楚、及时合并就不乱。 |
| 冲突说明做错了 | 冲突只是多人改到同一位置，需要人工判断保留什么。 |
| PR 只是提交代码 | PR 也是讨论、审查、记录决策的地方。 |

## 典型场景

### 场景一：自己维护仓库

日常流程通常是：

```bash
git status
git add README.md
git commit -m "docs: update readme"
git push
```

先养成每次提交前看 `git status` 的习惯。它会告诉你哪些文件改了、哪些文件还没暂存。

### 场景二：用 VS Code 提交

如果你不想一开始就背命令，可以先用 VS Code 的源代码管理面板：

1. 修改文件。
2. 打开左侧“源代码管理”。
3. 查看变更。
4. 点击 `+` 暂存。
5. 写提交信息。
6. 点击提交并同步。

这个流程和命令行本质相同，只是把 `add`、`commit`、`push` 做成了按钮。

### 场景三：参与别人的开源项目

推荐流程是：

1. Fork 对方仓库到自己账号。
2. Clone 自己的 Fork。
3. 创建分支。
4. 修改并提交。
5. Push 到自己的 Fork。
6. 从 GitHub 页面发起 Pull Request。

新手第一次贡献时，优先选择文档错别字、README 补充、示例修复这类小修改。

## 少量自查

- 你能解释 `add` 和 `commit` 的区别吗？
- 你能判断一次修改是否应该新建分支吗？
- 你能说出 PR 里应该写哪些信息吗？
- 遇到冲突时，你知道先打开哪个文件吗？

## 外部资源

- [GitHub Docs](https://docs.github.com/)：GitHub 官方文档，适合查 Pull Request、Fork、Issues、Actions 等功能。
- [Pro Git Book](https://git-scm.com/book/en/v2)：系统学习 Git 的免费电子书。
- [GitHub Desktop Docs](https://docs.github.com/en/desktop)：适合不想一开始使用命令行的新手。
- [VS Code Source Control](https://code.visualstudio.com/docs/sourcecontrol/overview)：学习 VS Code 图形化 Git 操作。

## 下一步

如果你已经能提交和推送，下一步建议重点练 [分支](docs/part3-teamwork/3.1-branching.md) 和 [Pull Request](docs/part3-teamwork/3.2-pull-request.md)。团队协作真正开始变顺，通常就是从理解这两件事开始的。
