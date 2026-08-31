# Python 现代项目结构与 `pyproject.toml` 入门

刚学 Python 时，一个 `hello.py` 就足够。随着代码变多，你会开始遇到“模块找不到”“依赖装乱了”“换一台电脑跑不起来”等问题。这篇教程把单文件脚本逐步整理成一个结构清楚、可安装、可测试的小项目。

## 适合人群

- 已经会写函数和导入模块，准备做多文件项目的新手
- 会用虚拟环境，但还在用零散命令记录依赖的人
- 想看懂开源项目里的 `pyproject.toml`、`src/` 和 `tests/` 的人
- 准备把命令行脚本整理成可维护工具的人

## 学习目标

读完后，你应该能：

- 判断什么时候继续用单文件，什么时候建立项目结构
- 理解发行项目名、导入包名和模块名的区别
- 用 `pyproject.toml` 描述项目和依赖
- 使用 `src` 布局、可编辑安装和依赖组
- 知道哪些内容属于标准，哪些行为取决于 pip 或构建工具版本

## 单文件、模块、包和项目

这些词很像，但层级不同：

| 名称 | 示例 | 含义 |
| :--- | :--- | :--- |
| 脚本 | `hello.py` | 可以直接运行的 Python 文件 |
| 模块 | `formatter.py` | 可以被其他代码 `import` 的单个文件 |
| 导入包 | `study_tool/` | 含多个模块、通常带 `__init__.py` 的目录 |
| 发行项目 | `study-tool` | 可由 pip 安装的项目，对应 `pyproject.toml` 中的 `name` |
| 仓库 | GitHub 上的整个目录 | 还可以包含文档、测试、工作流等内容 |

发行项目名可以使用短横线，例如 `study-tool`；Python 导入包名通常使用下划线，例如 `study_tool`：

```python
from study_tool.core import greet
```

## 什么时候需要项目结构

继续使用单文件更合适的情况：

- 代码只有几十行
- 只运行一次或用途非常单一
- 没有第三方依赖或测试需求

建议升级为项目的信号：

- 出现多个 `.py` 文件并且互相导入
- 需要第三方库
- 想添加自动测试
- 需要让别人安装或复用
- 配置、文档和代码已经混在同一层

## 推荐的最小目录

下面采用 Python Packaging User Guide 介绍的 `src` 布局：

```text
study-tool/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── study_tool/
│       ├── __init__.py
│       └── core.py
└── tests/
    └── test_core.py
```

`src` 布局把“可导入代码”和仓库根目录里的配置、文档分开。它要求先安装项目再导入代码，多一步操作，却能更早暴露打包遗漏和错误导入路径。

## 第一步：创建虚拟环境

在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip --version
```

Linux 或 macOS 使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip --version
```

如果 PowerShell 阻止激活脚本，请回到 [Python 环境与依赖管理指南](ENVIRONMENT_GUIDE.md) 查看执行策略和解释器排查。不要用管理员权限把所有包安装到系统 Python 来绕过问题。

## 第二步：编写最小 `pyproject.toml`

`pyproject.toml` 是现代 Python 项目的统一配置入口。一个适合本地学习项目的最小示例如下：

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "study-tool"
version = "0.1.0"
description = "A small project for learning Python packaging"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []

[dependency-groups]
test = [
  "pytest>=8",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

三个核心区块分别负责：

- `[build-system]`：告诉构建前端使用哪个后端构建项目。
- `[project]`：保存项目名称、版本、Python 版本和运行时依赖等标准元数据。
- `[tool.*]`：保存 pytest、格式化器、静态检查器等工具自己的配置。

`[dependency-groups]` 是用于开发、测试、文档等内部依赖的标准区块。它不会自动变成发布包的运行时元数据。

## 第三步：编写包和测试

`src/study_tool/core.py`：

```python
def greet(name: str) -> str:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("name must not be empty")
    return f"Hello, {cleaned_name}!"
```

`src/study_tool/__init__.py`：

```python
from .core import greet

__all__ = ["greet"]
```

`tests/test_core.py`：

```python
import pytest

from study_tool import greet


def test_greet() -> None:
    assert greet("Ada") == "Hello, Ada!"


def test_greet_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        greet("   ")
```

测试导入的是已安装的 `study_tool`，这正是 `src` 布局想验证的事情：项目必须真的能被正确安装和导入。

## 第四步：可编辑安装

开发时在项目根目录运行：

```powershell
python -m pip install -e .
```

`-e` 表示 editable install，可编辑安装不会把源码复制成一份固定副本。你修改 `src/` 下的 Python 文件后，通常可以直接重新运行；修改项目名、入口脚本等元数据后，可能需要重新安装。

部署或持续集成环境更适合普通安装：

```powershell
python -m pip install .
```

普通安装更接近最终用户得到的效果，所以项目准备发布前，两种安装方式都应该验证。

## 第五步：安装测试依赖

pip 从 25.1 开始支持安装标准依赖组。先查看版本：

```powershell
python -m pip --version
```

pip 25.1 或更新版本可以运行：

```powershell
python -m pip install --group test
python -m pytest
```

如果旧版 pip 报“不认识 `--group`”，可以在当前虚拟环境中更新 pip：

```powershell
python -m pip install --upgrade pip
```

如果项目必须兼容旧工具，也可以暂时保留 `requirements-dev.txt`。依赖组是标准的数据结构，但具体安装命令仍取决于所用工具及其版本。

## 三类依赖不要混用

| 写在哪里 | 谁需要 | 示例 |
| :--- | :--- | :--- |
| `[project].dependencies` | 安装项目后运行代码的所有用户 | `requests`、`pandas` |
| `[project.optional-dependencies]` | 用户主动选择的额外功能 | `excel`、`plot` |
| `[dependency-groups]` | 开发者测试、写文档、检查代码 | `pytest`、文档工具 |

例如，只有导出 Excel 时才需要 `openpyxl`，可以定义额外功能：

```toml
[project.optional-dependencies]
excel = ["openpyxl>=3.1"]
```

用户安装时选择：

```powershell
python -m pip install ".[excel]"
```

不要把 pytest 之类只在开发时使用的工具塞进运行时依赖，否则普通用户也会安装一批用不到的包。

## `requirements.txt` 还要不要用

`pyproject.toml` 并不意味着 `requirements.txt` 完全消失：

- 库项目通常把直接运行依赖写在 `[project].dependencies`。
- 应用部署可能还需要锁定后的依赖清单，以获得可重复环境。
- 一些旧平台和部署系统只识别 `requirements.txt`。

新手先分清“声明我直接依赖什么”和“锁定本次部署的完整版本”是两件事。Python 标准定义了项目元数据和依赖组，但没有规定所有工具必须使用同一种锁文件。

## 常见错误

### 在仓库根目录能导入，换目录就失败

这通常说明你依赖了“当前目录恰好在导入路径里”。执行可编辑安装，再从其他目录测试导入。

### 项目名和导入名混淆

安装命令可能使用 `study-tool`，导入语句则是 `import study_tool`。先看 `src/` 下真实包目录，不要只猜 PyPI 名称。

### 虚拟环境已经激活，VS Code 仍然报模块不存在

终端环境和编辑器解释器可能不是同一个。使用 `python -c "import sys; print(sys.executable)"` 查看终端解释器，再在 VS Code 中选择相同路径。

### 把 `.venv` 提交进仓库

虚拟环境体积大且与本机有关，应把 `.venv/` 写进 `.gitignore`，在新电脑上根据项目配置重新创建。

### 一开始就追求发布到 PyPI

先保证本地安装、导入和测试全部通过。发布还涉及许可证、版本管理、构建产物和账号安全，可以在项目稳定后再学。

## 完成检查清单

- [ ] 项目在独立虚拟环境中运行
- [ ] 可导入代码位于 `src/包名/`
- [ ] `pyproject.toml` 同时包含 `[build-system]` 和 `[project]`
- [ ] 运行依赖、可选功能和开发依赖没有混在一起
- [ ] `python -m pip install -e .` 可以完成
- [ ] 从测试中能正常导入包
- [ ] `.venv/`、缓存和构建产物没有进入 Git

## 官方资料

- [Python Packaging User Guide：编写 `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Python Packaging User Guide：`src` 布局与平铺布局](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [Python Packaging User Guide：Dependency Groups 规范](https://packaging.python.org/en/latest/specifications/dependency-groups/)
- [pip：本地项目和可编辑安装](https://pip.pypa.io/en/stable/topics/local-project-installs/)
- [pip：安装 Dependency Groups](https://pip.pypa.io/en/stable/user_guide/#dependency-groups)

## 下一步

先把一个现有的两三文件小工具按本页结构整理好，再回到 [Python 小项目学习路线](PROJECT_ROADMAP.md) 继续增加命令行参数、文件处理和测试。不要一次迁移所有脚本，小项目跑通一遍后再复用结构。
