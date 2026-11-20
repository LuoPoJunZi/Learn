# A2A 多智能体协作协议入门：Agent Card、任务状态机与安全边界

## 适合人群与学习目标

适合已经读过 [MCP 工具协议入门](MCP_TOOLS_GUIDE.md) 和 [Agent 工具调用案例](examples/ai_agent_tool_calling.py) 的学习者。

资料核验：**2026-11-20**，参考 A2A（Agent-to-Agent）开放协作协议核心概念。完成后你应该能够：

- 区分 **MCP**（Agent ↔ 工具/数据源，垂直层）与 **A2A**（Agent ↔ Agent，水平层）的职责分工。
- 看懂 **Agent Card**（能力名片）如何用于跨团队、跨框架的能力发现。
- 理解任务生命周期状态机（`submitted → working → completed / failed / canceled`）以及终态不可逆原则。
- 识别多智能体协作中的核心安全风险（不要把远端 Agent 的回复直接当成受信任的系统指令）。

---

## MCP 和 A2A 如何分工协作？

```text
用户
 │
 ▼
[协调 Agent (Client Agent)]
 ├── 垂直层 (MCP) ──> 本地文件、数据库、计算器、API 工具
 └── 水平层 (A2A) ──> [远端专家 Agent (Remote Agent)]
                         └── 垂直层 (MCP) ──> 远端专属数据源
```

- **MCP** 解决“一个 Agent 怎么安全调用工具、读取资源”。
- **A2A** 解决“多个彼此独立、甚至由不同框架开发的 Agent 怎么发现彼此、委派子任务、跟踪进度并交接产物（Artifacts）”。

---

## 核心概念三步走

### 1. Agent Card：先看名片再派活

真实 A2A 服务通常在约定路径（如 `/.well-known/agent.json`）暴露一份 JSON 名片，声明：
- `name`、`description`、`version`
- `capabilities`（是否支持流式推送等）
- `skills`（暴露哪些具体技能，例如本例中的 `word-count` 和 `extract-title`）

客户端 Agent 先读取名片，确认目标支持所需技能，再发起委派，避免盲目猜测。

### 2. 任务生命周期状态机

跨 Agent 任务往往不是瞬间完成的，因此协议用显式状态跟踪：

```text
submitted ──> working ──> completed (终态)
    │            │
    │            ├──> failed    (终态)
    └──> canceled <──┘          (终态)
```

- 一旦进入终态（`completed`、`failed`、`canceled`），任务不能再被回退或重复取消。
- 本仓库示例显式校验状态转移：对已完成的 `task-101` 发起 `tasks/cancel` 会被拒绝并返回错误码 `-32002`。

### 3. 区分“协议错误”与“任务执行失败”

| 场景 | 本例表达方式 | 含义 |
| :--- | :--- | :--- |
| 未知方法 / 未知技能 / 重复 ID | 顶层 `error`（`-32601` / `-32602`） | 请求本身不合法或无法路由 |
| 查询或取消不存在的任务 | 顶层 `error`（`-32001`） | 目标任务不存在 |
| 对终态任务再次取消 | 顶层 `error`（`-32002`） | 违反状态机约束 |
| 技能识别成功，但输入空白文本 | `result.status.state = "failed"` | 任务本身运行失败，保留状态历史便于审计 |

---

## 运行与验证

示例要求 Python 3.10+，仅使用标准库，不监听端口、不发起网络请求：

```powershell
python "Python\Modern AI\examples\a2a_agent_protocol.py"
python -m unittest discover -s scripts/tests -p "test_frontier_ai_2026.py" -v
```

---

## 多智能体协作的安全边界

1. **把远端 Agent 的输出当作不可信数据**：另一个 Agent 返回的文本里可能混入恶意提示词注入，绝不能不经校验直接拼接为高权限工具参数。
2. **最小权限委派**：不要把当前用户的全部 API Key 或完整对话历史一股脑转发给子 Agent，只传递完成该子任务所需的最小上下文。
3. **显式超时与可取消**：真实网络调用必须设置超时上限，并支持通过 `tasks/cancel` 及时中止失控子任务。

---

## 常见误区与实现边界

- **实现边界**：[a2a_agent_protocol.py](examples/a2a_agent_protocol.py) 是内存中的离线教学模拟器，用于演示 Agent Card 副本保护、JSON-RPC 风格请求分发、状态机转移与错误码分层；它**不是**可直接对外提供服务的 HTTP/SSE 服务器，也不包含 OAuth2/mTLS 认证或真实 LLM 推理。
- **规范演进**：A2A 协议仍在持续迭代，真实生产接入请以官方最新规范与 SDK 为准。

---

## 权威参考与下一步

- [A2A Protocol 官方网站与规范](https://a2a-protocol.org/)
- [MCP 工具协议与安全边界](MCP_TOOLS_GUIDE.md)
- [Agent 评测入门](AGENT_EVALUATION_GUIDE.md)
- [组相对策略优化 GRPO 入门](GRPO_GUIDE.md)
