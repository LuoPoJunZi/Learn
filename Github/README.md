# GitHub 新手入门教程

这是一套写给 GitHub 新手的教程。它不默认你会命令行，也不默认你理解“仓库、分支、提交、PR”这些词。你可以把它当成一本从 0 到能参与团队协作的小册子。

## 适合谁阅读

- 从来没用过 Git 或 GitHub 的同学
- 只会下载代码，但不知道怎么上传代码的人
- 想用 VS Code 提交代码、不想一直背命令的人
- 准备加入团队项目，需要理解分支、Pull Request、冲突解决的人

## 推荐阅读路线

如果你是纯新手，建议按顺序读：

1. [第一部分：基础概念与准备工作](docs/part1-basics/1.1-what-is-git.md)
2. [第二部分：Git 基础操作](docs/part2-git-cmds/2.1-core-concepts.md)
3. [第三部分：团队协作核心工作流](docs/part3-teamwork/3.1-branching.md)
4. [第四部分：团队规范与协作礼仪](docs/part4-conventions/4.1-gitignore-guide.md)
5. [第五部分：扩展应用与效率技巧](docs/part5-advanced-tools/5.1-url-shortener.md)

如果你只想解决一个具体问题，可以直接跳到下面的章节。

## 教程目录

### 第一部分：基础概念与准备工作

- [1.1 Git 与 GitHub 到底有什么区别？](docs/part1-basics/1.1-what-is-git.md)
- [1.2 账号注册与安装 Git](docs/part1-basics/1.2-setup.md)
- [1.3 配置 SSH 密钥](docs/part1-basics/1.3-ssh-keys.md)

### 第二部分：Git 基础操作

- [2.1 工作区、暂存区、本地仓库](docs/part2-git-cmds/2.1-core-concepts.md)
- [2.2 日常指令备忘录](docs/part2-git-cmds/2.2-daily-commands.md)
- [2.3 把本地已有文件夹推送到 GitHub](docs/part2-git-cmds/2.3-upload-local-project.md)
- [2.4 用 VS Code 丝滑提交](docs/part2-git-cmds/2.4-vscode-workflow.md)

### 第三部分：团队协作核心工作流

- [3.1 什么是分支？](docs/part3-teamwork/3.1-branching.md)
- [3.2 如何发起和合并 Pull Request](docs/part3-teamwork/3.2-pull-request.md)
- [3.3 代码冲突怎么办？](docs/part3-teamwork/3.3-merge-conflict.md)

### 第四部分：团队规范与协作礼仪

- [4.1 .gitignore 配置规则](docs/part4-conventions/4.1-gitignore-guide.md)
- [4.2 Commit 提交规范](docs/part4-conventions/4.2-commit-message.md)
- [4.3 如何提一个高质量 Issue](docs/part4-conventions/4.3-issues.md)

### 第五部分：扩展应用与效率技巧

- [5.1 GitHub 脚本域名短链化与国内加速访问](docs/part5-advanced-tools/5.1-url-shortener.md)

## 学习建议

Git 学习最容易卡住的地方，不是命令太多，而是脑子里没有“文件从哪里到哪里”的地图。建议你在学习时准备一个测试文件夹，边读边操作，不要只看。

每次学习一个新命令时，先问自己三个问题：

- 它会影响工作区、暂存区、本地仓库，还是远程仓库？
- 它会不会改动我的文件内容？
- 如果做错了，能不能撤回？

## 贡献

如果你发现内容有错误、步骤过时、表达不清楚，欢迎参考 [CONTRIBUTING.md](CONTRIBUTING.md) 参与改进。

