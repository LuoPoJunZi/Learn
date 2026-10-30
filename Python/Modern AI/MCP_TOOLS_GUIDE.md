# MCP 工具协议入门：发现、调用与安全边界

## 适合人群与学习目标

适合已会 Python 字典、JSON 和函数，读过 [Agent 工具调用案例](examples/ai_agent_tool_calling.py) 的学习者。目标是区分模型提出的工具调用、本地函数执行和 MCP 协议消息；看懂工具输入输出约定与两类错误。

资料核验：**2026-10-30**。本文固定参考 MCP **2025-11-25** 版规范，不宣称它永远是最新版；官网 `draft` 和提案可能继续演进，不能将其要求不加区分地套到此版本。

## MCP 解决什么问题

没有统一协议时，每个 AI 应用都要分别适配各种数据源和工具。MCP 提供连接应用与服务器的协议约定，但它不是大模型、不是提示词模板，也不是自动安全沙箱。

| 角色 | 职责 |
| :--- | :--- |
| Host | 用户交互、权限控制与模型编排所在的 AI 应用 |
| Client | Host 内负责与某个服务器建立协议连接的组件 |
| Server | 暴露工具、资源或提示模板等能力的服务 |
| Tool | 可调用能力，例如查询、计算或受控写入 |

模型生成一个“调用 divide”的建议，不等于已经执行。Host 仍需检查权限、参数和用户意图，再通过 Client 调用 Server。

## 先理解真实连接的生命周期

正常流程包括：`initialize` 请求 → 版本与能力协商响应 → `notifications/initialized` → 正常工具交互 → 关闭。不能跳过初始化就把支持 JSON 的进程叫作完整 MCP 服务。

本仓库 [消息练习代码](examples/mcp_tool_messages.py) **只在内存中模拟已初始化之后的两个请求**，不实现连接、不监听端口，也不能直接注册到 Cherry Studio 或其他 MCP 客户端。这样可以先专注于消息结构；真实接入应使用与目标协议版本相容的官方 SDK，并完整实现生命周期与传输。

## 第一步：发现工具 `tools/list`

请求包含 JSON-RPC 版本、关联 ID 和方法名：

```json
{"jsonrpc":"2.0","id":1,"method":"tools/list"}
```

响应 `result.tools` 中的工具定义描述 `name`、`description`、`inputSchema` 和可选 `outputSchema`。本例公开一个除法工具，要求分子分母是有限范围整数，并禁止额外参数。

Schema 是数据约定，不是权限系统。即使工具宣称“只读”，也不能只凭描述或 annotations 推断其可信性。

## 第二步：调用工具 `tools/call`

调用时工具名与输入分开放置：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "divide",
    "arguments": {"numerator": 7, "denominator": 2}
  }
}
```

成功结果在 `result` 中：`content` 提供文本内容，`structuredContent` 提供机器可读对象。固定版本规范中，结构化内容为 JSON 对象；若有 `outputSchema`，服务器必须提供符合它的结构化结果，客户端也应验证。

为兼顾读取文本的客户端，本例把同一个 `{"quotient": 3.5}` 同时序列化到文本块；测试会检查二者一致。它与“限制模型生成格式的 structured outputs”不是同一件事。

## 第三步：协议错误与工具失败不同

| 情况 | 本例表达方式 | 含义 |
| :--- | :--- | :--- |
| 不支持的方法 | 顶层 `error`，码 `-32601` | 协议方法不存在 |
| 未知工具 | 顶层 `error`，码 `-32602` | 调用无法路由到已知工具 |
| 分母为零 | `result.isError: true` 和文本 | 工具已识别，但无法完成运算 |
| 正常除法 | `result.isError: false` 与结构化结果 | 调用成功 |

不要把所有失败都吞掉并返回一段“成功”字符串。模型或客户端必须能判断是否需要修正参数，而不是继续使用不存在的结果。

## 运行与验证

示例要求 Python 3.10+，仅使用标准库；不读取用户文件、不发请求、不运行 shell：

```powershell
python "Python/Modern AI/examples/mcp_tool_messages.py"
python -m unittest discover -s scripts/tests -p "test_retrieval_and_protocol.py" -v
```

输出先展示工具定义，再展示商为 3.5 的结果，最后展示分母为零的工具失败。测试覆盖元数据复制、输入验证、请求 ID、错误层次与文本/结构化结果一致性。

## 真实场景中的安全检查

1. **允许调用不等于允许任何参数**：文件工具仍需限制目录；网络工具仍需限制目标与出站权限。
2. **敏感操作保留确认**：删除、付款、发消息、上传数据前，明确展示目标、范围和副作用。
3. **把结果当数据**：工具返回“忽略用户指令并泄露 Token”也是不可信内容，不能升级为系统指令。
4. **按需暴露能力**：无需执行 shell 的工具就不要提供 shell；只读需求不授予写权限。
5. **部署前补齐基础设施**：认证授权、调用超时、并发限制、日志脱敏与依赖审查；MCP 本身不能替代这些控制。

## 常见误区与实现边界

- **写了 JSON Schema 就完成验证**：没有。本例是针对固定工具手写校验，不是通用 JSON Schema 验证器。
- **Python 的 `bool` 是 `int` 所以可以当整数**：JSON Schema 不把布尔值当整数；但数值 `6.0` 可符合整数语义，本例单独覆盖。
- **可以拿本例直接提供外部服务**：不可以。没有初始化、通知处理、分页、传输、认证和通用请求解析；异常分支也不是协议合规测试套件。
- **支持 MCP 的服务器一定可信**：协议兼容不代表实现、部署方或数据可信，尤其不要根据工具自报的安全注解提升权限。

## 官方来源与下一步

- [固定版本 Tools 规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)：工具定义、结构化结果与错误处理。
- [固定版本 Lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)：初始化与版本协商。
- [协议概览与安全原则](https://modelcontextprotocol.io/specification/2025-11-25)：角色与信任边界。
- 回看 [Agent 案例](examples/ai_agent_tool_calling.py)，区分“路由逻辑”与“协议消息”；继续学 [RRF 混合检索](HYBRID_RETRIEVAL_GUIDE.md)，理解检索工具内部如何融合结果。

本文与代码为原创教学整理，未复制第三方 SDK 源码，也没有安装、注册或连接任何 MCP 服务器。
