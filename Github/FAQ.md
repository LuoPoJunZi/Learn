# GitHub 常见问题

这份 FAQ 收集 Git 和 GitHub 新手最容易卡住的问题。遇到报错时，先看 `git status`，再对照下面的问题定位。

## 我已经 commit 了，为什么 GitHub 网页上看不到？

`commit` 只是保存到本地仓库，还需要 `push` 到远程仓库。

常见流程：

```bash
git status
git add README.md
git commit -m "docs: update readme"
git push
```

如果 `git push` 报错，重点看报错里是否出现 `rejected`、`authentication`、`permission`、`non-fast-forward`。

## `git add .` 和 `git add 文件名` 有什么区别？

`git add .` 会暂存当前目录下的大多数改动，适合你确认所有改动都要提交时使用。

`git add 文件名` 只暂存指定文件，更适合新手，因为不容易把临时文件、缓存文件、账号配置一起提交。

建议新手先用：

```bash
git status
git add README.md
git status
```

确认暂存内容后再提交。

## 提交信息应该怎么写？

不要只写 `update`。一条好的提交信息应该说明这次改动的类型和目的。

常见写法：

```text
docs: update linux learning guide
fix: correct broken markdown link
feat: add python beginner examples
```

新手不必追求复杂规范，先做到“别人看提交历史能知道你改了什么”。

## 什么时候需要新建分支？

如果只是自己仓库里改一个错别字，可以直接在当前分支提交。

如果是以下情况，建议新建分支：

- 修改内容较多。
- 需要别人 Review。
- 可能会改坏现有内容。
- 同时有多个方向在推进。

分支的价值是隔离风险，让主线保持稳定。

## Pull Request 是给谁看的？

Pull Request 不是只给 GitHub 看的，它主要给人看。

一个好的 PR 应该说明：

- 这次改了什么。
- 为什么要改。
- 怎么验证。
- 有没有风险或未完成事项。

如果 PR 是文档修改，也可以写“已运行文档检查脚本”。

## Fork 和 Branch 有什么区别？

Branch 是同一个仓库里的分支。

Fork 是把别人的仓库复制到自己的账号下。参与开源项目时，你通常没有权限直接推送到别人仓库，所以需要先 Fork。

简单记忆：

- 自己仓库内开发：Branch。
- 给别人仓库贡献：Fork + Branch + PR。

## 冲突是不是说明我做错了？

不是。冲突只是说明两边修改了同一段内容，Git 不知道该自动保留哪一边。

处理冲突时不要慌：

1. 打开冲突文件。
2. 找到 `<<<<<<<`、`=======`、`>>>>>>>`。
3. 判断最终应该保留什么。
4. 删除冲突标记。
5. 保存、`git add`、继续提交或合并。

## 我不想用命令行，可以只用 VS Code 吗？

可以。VS Code 的源代码管理面板可以完成查看变更、暂存、提交、同步等操作。

但建议至少理解这些命令的含义：

- `git status`
- `git add`
- `git commit`
- `git push`
- `git pull`

理解命令不是为了天天手敲，而是为了出错时知道发生了什么。

## 下一步

如果这些问题仍然不清楚，建议回到 [GitHub 学习导读](LEARNING_GUIDE.md)，再按 [教程目录](README.md) 从基础概念重新走一遍。
