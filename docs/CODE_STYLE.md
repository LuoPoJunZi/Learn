# 示例代码规范

这份规范用于统一仓库中新写或正在维护的 Python、HTML、CSS、JavaScript、PowerShell 和 Matlab 示例。旧案例不要求一次性重写，但修改旧代码时应逐步靠近本规范，并优先保证示例仍可运行、原有内容仍可查找。

## 适合人群

- 准备新增示例代码的学习者。
- 修复或整理现有脚本的维护者。
- 想在提交前检查代码质量的贡献者。

## 通用约定

- 文本文件使用 UTF-8 编码和 LF 换行。
- 文件末尾保留一个换行，不保留行尾空格。
- 变量名应表达用途，避免只有 `a`、`data1`、`temp2` 这类难以理解的名称。
- 注释解释“为什么这样做”或风险，不重复翻译代码本身。
- 示例中的网址、路径、账号和密钥必须是明显的占位值，真实凭证只能从环境变量或安全配置中读取。
- 删除、移动、覆盖、批量发送等操作应提供预演模式，或在执行前要求用户明确确认。

仓库根目录的 [.editorconfig](../.editorconfig) 和 [.gitattributes](../.gitattributes) 会帮助编辑器保持编码、缩进和换行一致。

## Python

- 推荐 Python 3.10 或更新版本，使用 4 个空格缩进。
- 函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_CASE`。
- 使用明确导入，禁止 `from module import *`。
- 可复用逻辑放进函数；可执行入口放进 `main()`，并使用 `if __name__ == "__main__":` 保护。
- 文件路径优先使用 `pathlib.Path`；读取文本时明确写出 `encoding="utf-8"`。
- 网络请求必须设置超时，并捕获库提供的具体异常。
- 新增公共函数建议写参数和返回值类型；教学代码可保留直观写法，但不能牺牲安全性。
- 单行建议不超过 100 个字符，过长表达式应按逻辑拆行。

安全的脚本入口示例：

```python
from pathlib import Path


def count_lines(file_path: Path) -> int:
    text = file_path.read_text(encoding="utf-8")
    return len(text.splitlines())


def main() -> int:
    print(count_lines(Path("example.txt")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## HTML、CSS 和 JavaScript

- HTML 页面至少包含 `<!doctype html>`、`lang`、UTF-8、viewport 和非空 `title`。
- 优先使用 `header`、`nav`、`main`、`section`、`button` 等语义元素。
- 图片必须写 `alt`；纯装饰图片使用空值 `alt=""`。
- 按钮明确写 `type="button"` 或 `type="submit"`，交互状态同步更新必要的 ARIA 属性。
- CSS 使用 2 个空格缩进，类名使用小写短横线；不要把 HTML 注释 `<!-- -->` 写进 CSS。
- JavaScript 使用 `const` 和 `let`，避免隐式全局变量；事件逻辑拆成小函数并处理元素不存在的情况。

## PowerShell

- 使用 4 个空格缩进，脚本开头设置 `$ErrorActionPreference = 'Stop'`。
- 路径参数优先配合 `-LiteralPath`，避免通配符意外扩大操作范围。
- 外部命令执行后检查 `$LASTEXITCODE`，失败时给出可定位的文件或命令信息。
- 批量删除、移动或覆盖前，先解析并验证目标路径。

## Matlab

- 函数文件名必须与文件中的主函数名完全一致，包括大小写和拼写。
- 函数、变量使用有意义的名称；需要固定随机结果时显式调用 `rng(seed)`。
- 函数内部避免 `clear all`、`clc` 和 `close all`，这些命令会影响调用者工作区和调试过程。
- 尽量避免 `eval` 和 `global`；确实依赖旧算法实现时，应在 README 中说明原因和输入输出。
- 矩阵可直接预分配时不要在循环中反复扩展，重要维度在注释中说明。
- 算法目录应保留入口脚本、目标函数、辅助函数和对应 README，改名后同步更新文档链接。

## Wolfram Language

- 新案例使用 `.wl` 保存可运行源码，Notebook 主要用于交互展示和逐段实验。
- 自定义符号使用有意义的英文名称；运行独立脚本前用 `ClearAll` 清理会重复定义的符号。
- 内置函数使用方括号调用，列表使用花括号，方程使用 `==`，规则使用 `->`。
- 中间表达式不需要显示时使用分号；希望读者观察的最终图形或结果保留为最后一个表达式。
- 随机案例显式调用 `SeedRandom`，避免每次运行得到完全不同的教学结果。
- 导出文件、访问网络或调用云端能力时，必须在 README 中说明输出位置、网络需求和隐私边界。

## 提交前检查

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-all.ps1
```

统一入口会检查仓库结构、Python 语法和常见风险、JavaScript 语法、HTML 基础结构与常用属性、PowerShell 语法、Matlab 主函数名、文档链接、代码块和 Git 空白错误。Matlab 案例还应在安装了 Matlab 的电脑上运行 Code Analyzer，并用小规模参数执行入口脚本。

## 下一步入口

- [文档整理规范](STYLE_GUIDE.md)
- [文档维护者指南](../scripts/MAINTAINER_GUIDE.md)
- [Python 环境与依赖管理指南](../Python/ENVIRONMENT_GUIDE.md)
- [Matlab 示例运行指南](../Matlab/RUNNING_EXAMPLES.md)
