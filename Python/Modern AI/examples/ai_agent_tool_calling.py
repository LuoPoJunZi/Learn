"""Demonstrate AI agent tool-calling with a ReAct loop."""

from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
JsonDict = dict[str, object]

# ---------------------------------------------------------------------------
# Model Context Protocol (MCP) 概念说明
# ---------------------------------------------------------------------------
# MCP 被称为 "AI 的 USB-C"：它为 LLM 与外部工具之间定义了一套标准化
# 的 JSON-RPC 通信协议，让任何符合 MCP 规范的 Tool Server 都能被任何
# 兼容的 Agent 客户端直接调用，无需为每家 LLM 写定制集成。
#
# 核心流程：客户端发现 → 读取 tool schema → 构造 JSON 调用 → 接收结果
# 本脚本用纯 Python 模拟了这一 "schema → 调用 → 观察" 的数据流。
# ---------------------------------------------------------------------------


# ── 工具定义 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tool:
    """A callable tool with a JSON-schema-style parameter description."""

    name: str
    description: str
    parameters: JsonDict          # 简化版 JSON Schema

    def execute(self, arguments: JsonDict) -> str:
        """Dispatch to the concrete implementation by tool name."""
        dispatch = {
            "calculator": _run_calculator,
            "file_search": _run_file_search,
            "weather_lookup": _run_weather_lookup,
        }
        handler = dispatch.get(self.name)
        if handler is None:
            return f"错误：未知工具 '{self.name}'"
        return handler(arguments)


def _run_calculator(args: JsonDict) -> str:
    """Evaluate bounded arithmetic without executing Python source."""
    expression = args.get("expression", "")
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        return "错误：请输入不超过 200 个字符的算术表达式"

    operations = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
    }

    def calculate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = calculate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            result = operations[type(node.op)](calculate(node.left), calculate(node.right))
        else:
            raise ValueError("只支持数字、括号与 + - * / 运算")
        if not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("数值超出教学计算器范围")
        return result

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("表达式过于复杂")
        result = calculate(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"计算出错：{exc}"
    return f"计算结果：{result:g}"


def _run_file_search(args: JsonDict) -> str:
    """Simulate searching a knowledge base for a keyword."""
    keyword = str(args.get("keyword", ""))
    knowledge = {
        "transformer": "Transformer 架构由 Vaswani 等人于 2017 年提出，核心是自注意力机制。",
        "agent":       "AI Agent 通过感知-推理-行动循环自主完成任务，常配合工具调用。",
        "mcp":         "MCP（Model Context Protocol）是 Anthropic 提出的开放标准，统一 LLM 工具接口。",
    }
    for key, value in knowledge.items():
        if key in keyword.lower():
            return f"检索结果：{value}"
    return f"未找到与 '{keyword}' 相关的文档。"


def _run_weather_lookup(args: JsonDict) -> str:
    """Return simulated weather data for a city."""
    city = str(args.get("city", ""))
    forecasts: dict[str, str] = {
        "beijing":  "北京：晴，28°C，湿度 35%",
        "shanghai": "上海：多云，26°C，湿度 72%",
        "tokyo":    "东京：阵雨，22°C，湿度 85%",
    }
    return forecasts.get(city.lower(), f"暂无 '{city}' 的天气数据。")


# 工具注册表 —— 类比 MCP Server 的工具清单
TOOL_REGISTRY: list[Tool] = [
    Tool(
        name="calculator",
        description="对数学表达式求值",
        parameters={"expression": {"type": "string", "description": "算术表达式"}},
    ),
    Tool(
        name="file_search",
        description="在知识库中按关键词检索文档",
        parameters={"keyword": {"type": "string", "description": "搜索关键词"}},
    ),
    Tool(
        name="weather_lookup",
        description="查询指定城市的天气预报",
        parameters={"city": {"type": "string", "description": "城市名（英文）"}},
    ),
]

TOOLS_BY_NAME: dict[str, Tool] = {tool.name: tool for tool in TOOL_REGISTRY}


# ── Agent 状态与 ReAct 循环 ──────────────────────────────────────────────

@dataclass
class ToolCall:
    """A structured tool invocation produced by the agent."""

    tool_name: str
    arguments: JsonDict


@dataclass
class AgentState:
    """Mutable state that tracks the full Reason→Act→Observe history."""

    query: str
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_answer: str = ""


def _plan_tool_call(state: AgentState) -> ToolCall | None:
    """Simulate LLM reasoning: pick a tool based on simple keyword rules.

    真正的 LLM 会根据 system prompt 中提供的 tool schema 生成结构化
    JSON（即 function calling），这里用关键词匹配来模拟同样的协议流程。
    """
    query_lower = state.query.lower()

    # 已经调用过工具 → 不再调用，准备生成最终回答
    if state.observations:
        return None

    if any(op in query_lower for op in ("计算", "加", "减", "乘", "除", "多少")):
        expression = "".join(ch for ch in state.query if ch in "0123456789+-*/(). ")
        state.thoughts.append("用户需要数学计算，应调用 calculator 工具。")
        return ToolCall("calculator", {"expression": expression.strip()})

    if any(kw in query_lower for kw in ("天气", "weather", "气温")):
        city = "beijing"  # 简化：默认北京
        for c in ("shanghai", "tokyo", "beijing"):
            if c in query_lower:
                city = c
        state.thoughts.append(f"用户想查天气，应调用 weather_lookup 工具查询 {city}。")
        return ToolCall("weather_lookup", {"city": city})

    # 默认：当作知识检索
    state.thoughts.append("用户在提问，应调用 file_search 工具检索知识库。")
    return ToolCall("file_search", {"keyword": state.query})


def react_step(state: AgentState) -> bool:
    """Execute one Reason→Act→Observe cycle. Return True when done."""
    # ── Reason ──
    tool_call = _plan_tool_call(state)
    if tool_call is None:
        # 所有工具调用完毕，综合观察生成最终回答
        summary = "；".join(state.observations)
        state.final_answer = f"根据工具返回的信息——{summary}"
        state.thoughts.append("已获得足够信息，生成最终回答。")
        print(f"  [思考] {state.thoughts[-1]}")
        return True

    print(f"  [思考] {state.thoughts[-1]}")

    # ── Act ──
    state.tool_calls.append(tool_call)
    call_json = json.dumps(
        {"tool": tool_call.tool_name, "arguments": tool_call.arguments},
        ensure_ascii=False,
    )
    print(f"  [调用] {call_json}")

    # ── Observe ──
    tool = TOOLS_BY_NAME.get(tool_call.tool_name)
    if tool is None:
        observation = f"错误：工具 '{tool_call.tool_name}' 不存在"
    else:
        observation = tool.execute(tool_call.arguments)
    state.observations.append(observation)
    print(f"  [观察] {observation}")
    return False


def run_agent(query: str) -> str:
    """Run the full ReAct agent loop until a final answer is produced."""
    state = AgentState(query=query)
    print(f"\n{'─' * 60}")
    print(f"用户提问：{query}")

    max_steps = 5
    for _ in range(max_steps):
        done = react_step(state)
        if done:
            break

    if not state.final_answer:
        state.final_answer = "抱歉，在有限步数内未能完成推理。"

    print(f"  [回答] {state.final_answer}")
    return state.final_answer


# ── 入口 ─────────────────────────────────────────────────────────────────

def main() -> int:
    # 列出可用工具（类比 MCP 的 tools/list 请求）
    print("=== AI Agent 工具调用演示（ReAct 循环）===\n")
    print("已注册工具：")
    for tool in TOOL_REGISTRY:
        schema_str = json.dumps(tool.parameters, ensure_ascii=False)
        print(f"  • {tool.name}: {tool.description}  参数 {schema_str}")

    # 三个示例查询
    queries = [
        "请计算 (3 + 5) * 12 - 4",
        "Shanghai 的天气怎么样？",
        "什么是 Transformer？",
    ]
    for query in queries:
        run_agent(query)

    print(f"\n{'─' * 60}")
    print("演示结束。ReAct 循环：思考→调用→观察→回答，每一步数据均为结构化 JSON。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
