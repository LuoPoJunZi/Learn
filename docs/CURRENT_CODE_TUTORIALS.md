# 当前代码教程精选与阅读路线

本页用于把“找到新教程”变成“在仓库里真正学会并运行”。初始来源核验日期为 **2026-10-29**，新增专题核验日期为 **2026-10-30**；日期表示对应批次检查时间，不代表来源页面发布日期，也不保证外部链接永远不变。

优先选择官方维护的文档，配本仓库中文讲解和小示例，而不是按热度收集大量链接。下表区分了版本新增特性与持续适用的工程实践，**不是各语言最新稳定版本排行榜**。

## 适合谁、先学什么

- 刚入门：先学 [Python Basics](../Python/Basics/README.md) 或 [HTML Basics](../HTML/Basics/README.md)。
- 已经能写小脚本：先学 uv 管理环境，再学结构化并发。
- 已经会按钮事件和 `async/await`：直接运行可靠 Fetch 案例。
- 想了解语言变化：阅读下面的 Python 3.14 导读，按自己项目的目标版本选择，不必为了“追新”升级所有脚本。

## 2026-10-30 新增：协议、检索与组件样式

| 技术方向 | 官方来源 | 本仓库教程与代码 | 状态与边界 |
| :--- | :--- | :--- | :--- |
| MCP 工具协议 | [2025-11-25 Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)、[生命周期](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle) | [MCP 入门](../Python/Modern%20AI/MCP_TOOLS_GUIDE.md)，配离线工具消息与测试 | 固定版本，不混用 draft；不是完整服务器 |
| RAG 混合检索与 RRF | [Hybrid search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)、[RRF](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) | [RRF 中文教程](../Python/Modern%20AI/HYBRID_RETRIEVAL_GUIDE.md)，配融合与指标实现 | 成熟算法的当前工程应用，不宣称是今年新算法；手工排名不是实际语义模型 |
| CSS @scope | [MDN @scope](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/At-rules/@scope) | [组件样式边界案例](../HTML/Examples/scoped-css/README.md) | 检测实际解析与样式，保留回退；不声称所有旧浏览器可用 |

建议在现有 Agent/RAG 案例后学习前两项，在 CSS 选择器与继承基础后学习第三项。搜索也出现了 MCP draft 与不同语言 MDN 标签差异，本轮不将草案和单一 Baseline 年份当作稳定承诺。

## 2026-10-29 补充

| 主题 | 官方教程 | 仓库落地 | 最低要求与边界 |
| :--- | :--- | :--- | :--- |
| uv 项目管理 | [Working on projects](https://docs.astral.sh/uv/guides/projects/)、[Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) | [中文工作流教程](../Python/UV_WORKFLOW_GUIDE.md)，含应用与测试的完整编写步骤 | uv 持续更新；安装和同步可能联网，只在独立练习目录操作 |
| Python 结构化并发 | [协程与任务](https://docs.python.org/3.14/library/asyncio-task.html) | [讲解、源码与六项测试](../Python/Modern%20Python/README.md) | Python 3.11+；成熟 API，不是今年新发布；完全离线 |
| Fetch 错误与取消 | [Using Fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)、[AbortSignal](https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal) | [本地网页与离线测试](../HTML/Examples/reliable-fetch/README.md) | 现代浏览器；本地 HTTP 服务；测试用 Node 18+ |
| Python 3.14 语言变化 | [What's New](https://docs.python.org/3.14/whatsnew/3.14.html)、[模板字符串](https://docs.python.org/3.14/library/string.templatelib.html) | 本页下方的特性辨析与可复制片段 | 本节特定功能要求 3.14+，不将 3.14 宣称为当前最新版本 |

## Python 3.14：初学者值得先理解的变化

### 1. t-string 不是自动安全的 f-string

f-string 直接生成字符串；3.14 引入的 t-string 生成 `string.templatelib.Template`，保留静态部分和插值部分，交给后续处理器决定如何组合。

下面片段仅在 **Python 3.14+** 使用；不要粘到仍需兼容 3.10/3.13 的脚本里。旧解释器解析 `t"..."` 时就会报语法错误，写普通的运行时版本判断也不能解决同一个源文件的语法解析问题。

```python
name = "<Ada>"
plain = f"Hello, {name}!"
template = t"Hello, {name}!"

print(type(plain).__name__)       # str
print(type(template).__name__)    # Template
print(template.strings)          # ('Hello, ', '!')
print(template.interpolations[0].value)  # <Ada>
```

**关键边界**：t-string 不会自动完成 HTML 转义、SQL 参数化或 shell 转义。上面仅展示数据结构，不是安全模板引擎。数据库仍应使用驱动的参数绑定；网页输出使用适合上下文的转义或模板引擎；不要把用户输入拼接成可执行命令。

### 2. 类型注解的默认求值时机改变

3.14 默认延迟求值注解；读取注解的框架和工具应关注新的 `annotationlib`。这会影响运行时反射，不代表类型注解会自动检查函数参数。

```python
def double(value: int) -> int:
    return value * 2

print(double("ha"))  # 类型标注不会阻止这次调用；结果仍是 haha
```

静态类型检查与运行时校验是两回事。读取不可信对象的注解也不能当作安全操作：某些注解求值可能执行表达式。迁移框架或插件时，按官方说明选择注解读取方式并补测试，不要批量删除旧代码里的前向引用引号。

### 3. UUIDv7：排序友好的标识，不是密码

3.14 的标准库支持 `uuid.uuid7()`。它含有时间信息，适用于需要大致按生成时间排序的标识，但不应该作为认证 Token，也不能代替业务时间字段和全局事务顺序。

```python
import uuid

identifier = uuid.uuid7()  # Python 3.14+
print(identifier)
print(identifier.version)  # 7
print(identifier.time)     # Unix 毫秒时间戳
```

不要拿“看起来随机”推断安全性。认证秘密应使用专门的安全随机机制，例如标准库 `secrets`；生成标识与生成访问凭据是不同任务。详见 [uuid 官方参考](https://docs.python.org/3.14/library/uuid.html)。

### 4. 多解释器与自由线程不是零成本加速按钮

3.14 的标准库多解释器支持、自由线程构建等适合了解并行计算的人继续研究。不同方案有隔离、共享状态、扩展兼容与序列化约束；不是把同步大循环放进 `async def` 就自动多核并行。

初学者先通过 [结构化并发](../Python/Modern%20Python/README.md) 分清异步等待和 CPU 并行，再阅读 What's New 中的对应章节。不要以教程中的单个小例子推断真实业务性能。

## 一条能执行的学习路线

1. 看 [环境指南](../Python/ENVIRONMENT_GUIDE.md)，确认实际解释器版本与路径。
2. 按 [uv 教程](../Python/UV_WORKFLOW_GUIDE.md) 在仓库外编写应用与两个测试，理解锁文件。
3. 运行 [结构化并发案例](../Python/Modern%20Python/README.md)，观察成功、失败、超时各自的事件记录。
4. 运行 [可靠 Fetch](../HTML/Examples/reliable-fetch/README.md)，比较主动取消与旧结果抑制。
5. 按项目最低 Python 版本选择本页 3.14 特性；不满足版本时先学习概念，不改变现有运行环境。

## 如何判断下一篇教程是否值得加入

- **版本明确**：区分标准、实现、提案和浏览器实际支持，不把实验功能写成全面可用。
- **示例可验证**：列出环境、输入输出和测试方式；注明哪些是手动验证，哪些真正执行过。
- **有失败路径**：覆盖输入错误、异常、取消和超时，不只演示成功画面。
- **安全默认值**：不要求关闭浏览器安全、不泄露凭据、不对真实目录批量删除。
- **来源可追溯**：保留官方链接与用途说明，不复制来源不明的大型源码包。

本文为资料导读与原创教学整理。外部文档的许可和内容以各自站点为准；未将第三方项目源码整体引入本仓库。

返回 [docs 索引](README.md) 或 [仓库首页](../README.md)。
