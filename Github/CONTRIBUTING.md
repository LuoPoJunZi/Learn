# 参与贡献指南

感谢你愿意一起完善这套 GitHub 新手教程。这个项目的目标是：用普通人能听懂的话，把 GitHub 和团队协作讲清楚。

## 可以贡献什么

- 修正错别字、错链、格式问题
- 补充更清楚的操作截图说明
- 增加新手常见问题
- 改写不容易理解的段落
- 更新过时的 GitHub 页面或 Git 命令说明

## 写作原则

1. 面向新手：不要默认读者知道专业术语。
2. 先讲场景，再讲命令：告诉读者“为什么要这么做”。
3. 每个命令尽量配一句解释。
4. 避免炫技：本教程优先可理解、可复现。
5. 谨慎使用破坏性命令，比如 `reset --hard`、强制推送等。

## 推荐修改流程

1. Fork 本仓库到自己的账号。
2. 新建一个分支，例如：

```bash
git checkout -b docs/fix-typo
```

3. 修改文档。
4. 本地检查 Markdown 链接和排版。
5. 提交修改：

```bash
git add .
git commit -m "docs: fix typo in setup guide"
```

6. 推送到自己的 GitHub 仓库：

```bash
git push origin docs/fix-typo
```

7. 在 GitHub 页面发起 Pull Request。

## Commit 信息建议

推荐使用下面的格式：

```text
类型: 简短说明
```

常用类型：

- `docs`: 文档修改
- `fix`: 修复错误
- `style`: 调整格式，不改变内容含义
- `chore`: 杂项维护

示例：

```text
docs: add ssh key troubleshooting section
fix: correct git clone command
```

