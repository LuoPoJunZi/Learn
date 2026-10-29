# 现代 Python 实践：结构化并发

## 适合人群与目标

适合已理解函数、异常和列表，想学习异步任务管理的人。本案例使用 **Python 3.11+ 标准库**，无第三方依赖、网络请求或文件写入。目标是看懂并发任务的生命周期，而不是测试下载速度。

本页核验日期：2026-10-29。`TaskGroup` 和 `asyncio.timeout` 都在 Python 3.11 引入，属于现代推荐写法，**不是 2026 年新增功能**。

## 文件与运行

- [structured_concurrency.py](structured_concurrency.py)：成功、子任务失败、整批超时三种场景。
- [test_structured_concurrency.py](test_structured_concurrency.py)：顺序、清理、超时、外部取消和参数校验测试。

从仓库根目录运行示例和离线测试：

```powershell
python "Python/Modern Python/structured_concurrency.py"
python -m unittest discover -s "Python/Modern Python" -p "test_*.py" -v
```

预期成功结果是 `['SLOW', 'FAST']`；事件中 `fast` 先完成。失败和超时场景都应出现 `cleanup`。测试不对总耗时做硬性断言，避免把机器负载误判为功能失败。

## 先理解四个概念

| 概念 | 在案例里的含义 |
| :--- | :--- |
| 协程 | `async def` 定义的可暂停函数，调用后得到协程对象，不是立即完成的结果 |
| `await` | 等待异步操作，给事件循环调度其他任务的机会 |
| 任务 | 被事件循环调度的协程；本例通过 `group.create_task` 创建 |
| 结构化并发 | 子任务的生命周期由代码块约束，离开任务组前要处理完成或取消 |

并发不是 CPU 并行：多个任务在等待 I/O 时交替推进，不意味着纯 Python 大循环会自动分到多个核心。这里用 `asyncio.sleep` 模拟等待，避免向公共网站施加请求负载。

## 第一步：让生命周期有边界

核心逻辑如下，完整代码见上方文件：

```python
async with asyncio.timeout(timeout):
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(simulate_job(job, events)) for job in jobs]
return [task.result() for task in tasks]
```

任务组正常退出时，任务都已经结束，才可以安全读取 `result()`。按 `tasks` 的创建顺序读结果，所以完成顺序与返回顺序不同。

不要把下面写法误认为并发：

```python
for job in jobs:
    await simulate_job(job, events)
```

它会等一个任务结束才开始下一个。反过来，也不要创建一批任务后丢掉引用，完全不等它们结束。

## 第二步：一项失败怎么办

除取消等特殊情况外，任务组中一个任务抛出异常，会取消尚未结束的兄弟任务，等它们清理后以异常组形式抛出。本例用 `except* ValueError` 处理异常组中的对应类型；它与处理单一异常的 `except ValueError` 不同。

子任务用 `try/finally` 记录清理，即使等待被取消，也不会跳过 `finally`。真实场景通常使用 `async with` 管理连接与资源。不要为了让日志“好看”而吞掉 `asyncio.CancelledError`，那会破坏超时和任务组的取消机制。

## 第三步：超时放在哪里

本例的 `asyncio.timeout` 包住整批任务，所以限制的是**整批等待**，不是每个任务各自拥有完整超时。`TimeoutError` 在超时上下文外捕获。

取消是协作式的，不是强制终止线程或进程。任务必须有机会响应取消并清理，因此超时不应被理解为精确到毫秒的强制杀停保证。

真实批量请求还要考虑：

- 每次连接、读取的超时与整批超时是不同层次。
- 对大量任务应使用并发限额与有界队列，不能无限制创建任务。
- 本例一次只创建几个任务；它不是生产爬虫、重试器或限流器。

## 典型场景与常见错误

- **适合**：同时等待多个异步 API、处理多个独立异步输入。
- **不适合直接套用**：CPU 密集训练、大矩阵运算、百万级任务队列。
- **在协程里调用 `time.sleep`**：会阻塞事件循环，应使用异步库；已有阻塞 I/O 可以评估 `asyncio.to_thread`，但取消等待不等于强杀线程。
- **把同步 `requests.get` 放进 `async def` 就变异步**：不会。声明函数不改变内部库的阻塞性质。
- **Notebook 提示事件循环已运行**：Notebook 顶层使用 `await demo()`；命令行脚本入口才使用 `asyncio.run(demo())`。
- **Python 3.10 语法检查失败**：`except*` 与上述 API 需要 3.11+，不代表仓库全部内容都要求升级。

## 完成检查与下一步

能区分返回顺序与完成顺序、知道失败会影响兄弟任务，并且六项测试通过，才算跑通。下一步可以对照 [可靠 Fetch 教程](../../HTML/Examples/reliable-fetch/README.md)，比较 Python 与浏览器中的取消行为；环境管理见 [uv 工作流](../UV_WORKFLOW_GUIDE.md)。

## 官方资料与来源

- [Python 3.14：协程与任务](https://docs.python.org/3.14/library/asyncio-task.html)：任务组、超时、取消与 `to_thread`。
- [Python 3.11 新特性](https://docs.python.org/3.11/whatsnew/3.11.html)：核对任务组与异常组的引入版本。

讲解与代码为本仓库原创教学实现，依据官方 API 文档编写，未复制第三方项目源码。
