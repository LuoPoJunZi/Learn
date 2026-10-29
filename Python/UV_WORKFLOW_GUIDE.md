# uv 项目工作流：从依赖声明到可重复运行

## 适合人群与目标

适合已经会运行 Python 脚本、理解虚拟环境，想解决“换电脑就跑不起来”的学习者。读完后应能区分依赖声明、锁文件和实际环境，并使用 uv 运行测试。前置阅读：[环境指南](ENVIRONMENT_GUIDE.md)、[现代项目结构](MODERN_PROJECT_GUIDE.md)。

本页依据 2026-10-29 核验的 uv 官方文档整理。uv 是持续更新的第三方工具，不是 Python 标准库；这里介绍当前项目工作流，不宣称这些命令刚刚发布。

## 核心概念

| 文件或命令 | 解决的问题 |
| :--- | :--- |
| `pyproject.toml` | 声明项目、Python 版本范围和直接依赖 |
| `uv.lock` | 记录解析后的具体依赖版本与来源信息 |
| `.python-version` | 指定本地默认使用的解释器版本 |
| `.venv/` | 本机实际安装的环境，不应提交 |
| `uv lock` | 根据声明解析并更新锁文件，不等于安装环境 |
| `uv sync` | 根据锁文件同步环境，默认会精确同步、移除多余包 |
| `uv run` | 默认先检查锁文件并同步环境，再执行命令 |

类比：`pyproject.toml` 是购物需求，`uv.lock` 是确定好的清单，`.venv` 才是已经买回来的东西。只复制源码而不记录依赖，别人无法复原环境。

## 先确认工具，而不是直接执行远程安装脚本

在终端查看 uv 和 Python 版本：

```powershell
uv --version
python --version
```

没有 uv 时，按 [官方安装指南](https://docs.astral.sh/uv/getting-started/installation/) 选择适合系统的方法，确认安装源再执行。本文不要求安装 uv 才能阅读本仓库，也不会改造仓库根目录为 uv 项目。

以下命令供读者在**仓库外的新建练习目录**手动执行。初始化、添加依赖、同步和运行可能联网下载解释器或软件包，并写入项目目录与工具缓存。不要在现有重要项目上盲目练习。

## 第一步：创建独立应用项目

先创建一个不需要打包发布的练习项目，然后进入目录。`--bare` 避免依赖不同版本生成的默认示例文件：

```powershell
uv init --bare learn-uv-demo
cd learn-uv-demo
```

在目录里新建 `main.py`：

```python
def greet(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("name must not be empty")
    return f"Hello, {cleaned}!"


if __name__ == "__main__":
    print(greet("Ada"))
```

应用脚本不需要安装成包；需要发布可导入包时，再按 [现代项目结构](MODERN_PROJECT_GUIDE.md) 使用 `src` 和构建后端。

## 第二步：添加开发依赖与测试

`--dev` 将 pytest 放入开发依赖组，而不是运行依赖。该操作会更新项目配置、锁文件和环境：

```powershell
uv add --dev pytest
```

新建 `test_main.py`，正常输入和无效输入都要检查：

```python
import pytest

from main import greet


def test_greet() -> None:
    assert greet(" Ada ") == "Hello, Ada!"


def test_empty_name() -> None:
    with pytest.raises(ValueError):
        greet("   ")
```

运行应用与测试，不需要先手动激活虚拟环境：

```powershell
uv run python main.py
uv run python -m pytest
```

预期看到 `Hello, Ada!` 和两项测试通过。优先使用 `uv run python -m pytest`，让测试明确绑定当前项目解释器。

## 第三步：理解锁定与同步的区别

修改依赖声明后，用 `uv lock` 更新锁文件。想核对配置和锁文件是否一致，使用 `--locked`，不一致就报错而非偷偷改写：

```powershell
uv lock
uv sync --locked
uv run --locked python -m pytest
```

`--frozen` 则跳过“锁文件是否与声明一致”的检查，直接使用已有锁文件。它**不是**比 `--locked` 更严格的校验，初学者的提交前检查优先使用 `--locked`。

`uv sync` 默认精确同步，因此手动往 `.venv` 安装的额外包可能被移除。需要的依赖应通过 `uv add` 声明；不要把受管理环境当成随意堆包的全局环境。

## 典型场景：同学拿到项目

交付应用项目时保留源码、测试、`pyproject.toml` 和 `uv.lock`；使用 `.python-version` 时也一并保留。将 `.venv/`、`__pycache__/`、`.pytest_cache/` 加入该练习项目的 `.gitignore`。

接收者在项目目录执行：

```powershell
uv sync --locked
uv run --locked python -m pytest
```

锁文件提高可重复性，但不是安全审计，也不保证所有平台有相同的二进制包。解释器、操作系统、系统库和包索引权限仍会影响能否安装。

## 常见问题

- **已经有 pip 和 venv，还必须迁移吗？** 不必。理解现有工具优先；uv 是可选的环境和依赖工作流。
- **VS Code 提示找不到 pytest？** 选择项目 `.venv` 对应解释器，并核对终端是否在正确目录。
- **把 `uv.lock` 手动改一下可以吗？** 不建议。通过依赖声明和 uv 命令更新，再审查差异。
- **能直接 `uv run https://某地址/脚本.py` 吗？** 不要执行未经审查的远程代码；先获取并阅读源码、来源和副作用。
- **锁文件过期报错怎么办？** 确认配置变化符合预期，再在开发环境执行 `uv lock`，复测并审查差异，不要用 `--frozen` 掩盖问题。

## 完成检查

- 能说清声明、锁文件、实际环境分别存在哪里。
- 应用和两项测试都能通过 `uv run` 执行。
- `.venv` 没进入 Git，提交前审查了锁文件和配置差异。
- 没有为学习命令修改仓库中其他脚本的依赖环境。

## 官方资料与下一步

- [Working on projects](https://docs.astral.sh/uv/guides/projects/)：项目与文件职责。
- [Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)：自动更新、`--locked` 与 `--frozen`。
- [Managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)：运行依赖、可选依赖与开发组。
- 接着阅读 [结构化并发案例](Modern%20Python/README.md)，或回到 [Python 小项目路线](PROJECT_ROADMAP.md) 给已有脚本加测试。

本文为中文教学整理，代码为本仓库原创小示例；未搬运第三方项目源码。
