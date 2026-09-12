# 文档维护者指南

这个仓库以 Markdown 教程和示例说明为主。维护时最重要的是：新增内容能被找到，旧链接不失效，危险操作有提醒。

## 适合人群

- 准备新增教程或案例的人。
- 需要整理目录、更新 README 的维护者。
- 想在提交前确认文档没有明显问题的人。

## 维护流程

建议每次修改按这个顺序：

1. 明确本次只改哪些目录。
2. 新增文档时先确定入口位置。
3. 更新当前目录 `README.md`。
4. 如影响全局入口，更新根目录 `README.md` 和 `docs/LEARNING_PATH.md`。
5. 如是重要结构调整，更新 `CHANGELOG.md`。
6. 运行代码和文档检查脚本。
7. 查看 `git status --short`，确认没有临时文件。

## 文档检查脚本

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-docs.ps1
```

检查内容：

- Markdown 本地链接是否存在。
- 代码块是否闭合。
- 一级目录是否包含 `README.md`。

## 代码检查脚本

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-code.ps1
```

检查内容：

- Python 语法、行长、通配符导入、文本编码和网络请求超时。
- JavaScript 与 PowerShell 语法。
- HTML 页面的 doctype、语言、字符集、viewport、标题、重复 ID、图片 alt 和按钮类型。
- Matlab 主函数名是否与文件名完全一致。

详细规则见 [示例代码规范](../docs/CODE_STYLE.md)。Matlab Code Analyzer 的性能、未使用变量和旧 API 提示不会全部作为提交阻断项，应按案例逐步处理。

## 常见失败原因

| 现象 | 常见原因 | 处理方式 |
| :--- | :--- | :--- |
| Broken markdown links | 相对路径写错、文件改名、目录有空格未用尖括号 | 从当前 Markdown 文件位置重新计算路径。 |
| Unclosed code fences | 少写了结束的三个反引号 | 找到提示行，补上闭合代码块。 |
| Top-level directories without README | 新增一级目录但没写 README | 补一个目录说明和入口导航。 |

## 链接写法建议

普通路径：

```markdown
[Linux 学习导读](../Linux/LEARNING_GUIDE.md)
```

路径中有空格时，用尖括号：

```markdown
[神经网络](<../Python/Neural Network/README.md>)
```

外部链接要写清用途，不要只贴裸链接。

## 新增教程检查清单

一篇新手教程建议包含：

- 适合人群。
- 学习目标。
- 推荐阅读顺序。
- 核心概念。
- 典型场景。
- 常见误区或安全提醒。
- 下一步入口。

不是每篇都要很长，但至少要让读者知道：这篇文档解决什么问题，读完后去哪里。

## 提交前检查

```powershell
git status --short
git diff --stat
powershell -ExecutionPolicy Bypass -File .\scripts\check-all.ps1
```

如果出现缓存、临时文件、下载包、系统生成文件，不要一起提交。

统一检查也会验证必要根文件、一级目录 README、大小写或 Unicode 规范化后的路径冲突，以及已暂存和未暂存差异中的空白错误。GitHub Actions 会在推送和 Pull Request 中重复执行这套检查。
