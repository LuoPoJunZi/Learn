"""Demonstrate classical SSM vs selective state space models (Mamba core)."""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class ClassicalSSM:
    """A linear time-invariant (LTI) continuous/discrete state space model.

    h[t] = A * h[t-1] + B * x[t]
    y[t] = C * h[t]
    """

    state_dim: int
    a_diag: Vector  # Discrete transition factors; exp(negative rate) lies in (0, 1).
    b_weight: Vector  # Input projection
    c_weight: Vector  # Output projection

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or any(
            len(vector) != self.state_dim
            for vector in (self.a_diag, self.b_weight, self.c_weight)
        ):
            raise ValueError("State and parameter dimensions must match and be positive.")
        parameters = self.a_diag + self.b_weight + self.c_weight
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("SSM parameters must be finite.")

    def step(self, prev_state: Vector, input_val: float) -> tuple[Vector, float]:
        """Compute one step of linear recurrence in O(D) time."""
        if len(prev_state) != self.state_dim:
            raise ValueError("State dimension does not match the model.")
        new_state = [
            a * h + b * input_val
            for a, b, h in zip(self.a_diag, self.b_weight, prev_state)
        ]
        output_val = sum(c * h for c, h in zip(self.c_weight, new_state))
        return new_state, output_val

    def forward_sequence(self, inputs: Vector) -> list[float]:
        """Process an entire sequence of length L in O(L * D) time."""
        state = [0.0] * self.state_dim
        outputs: list[float] = []
        for x in inputs:
            state, y = self.step(state, x)
            outputs.append(y)
        return outputs


@dataclass(frozen=True)
class SelectiveSSM:
    """Mamba-style Selective State Space Model.

    Parameters delta, B, and C are functions of input x[t], breaking LTI constraints.
    """

    state_dim: int
    base_a: Vector

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or len(self.base_a) != self.state_dim:
            raise ValueError("State and transition dimensions must match and be positive.")
        if any(not math.isfinite(value) or value > 0 for value in self.base_a):
            raise ValueError("This decay example requires finite, non-positive rates.")

    def compute_selective_parameters(self, x: float) -> tuple[float, Vector, Vector]:
        """Make Delta (step size), B (input gate), and C (readout) data-dependent."""
        # Delta gating: softplus-like activation ensures positive discretization step
        raw_delta = 0.5 + 0.8 * math.tanh(x)
        delta = max(0.01, raw_delta)

        # Input-dependent B: decides HOW MUCH of x enters the hidden memory
        # If |x| is small (noise/filler), B approaches 0, actively ignoring filler tokens
        b_gate = [math.tanh(x * (i + 1.0)) for i in range(self.state_dim)]

        # Input-dependent C: modulates readout attention
        c_gate = [1.0 if abs(x) > 0.5 else 0.1 for _ in range(self.state_dim)]

        return delta, b_gate, c_gate

    def step(self, prev_state: Vector, input_val: float) -> tuple[Vector, float]:
        """Selective scan step: discretize A and B dynamically based on current x."""
        if len(prev_state) != self.state_dim:
            raise ValueError("State dimension does not match the model.")
        delta, b_t, c_t = self.compute_selective_parameters(input_val)

        new_state: list[float] = []
        for i in range(self.state_dim):
            # Exact exponential transition; delta * B is a first-order input approximation.
            a_bar = math.exp(delta * self.base_a[i])
            b_bar = delta * b_t[i]
            # State update
            h_new = a_bar * prev_state[i] + b_bar * input_val
            new_state.append(h_new)

        output_val = sum(c * h for c, h in zip(c_t, new_state))
        return new_state, output_val

    def forward_sequence(self, inputs: Vector) -> list[float]:
        """Process sequence with dynamic gating."""
        state = [0.0] * self.state_dim
        outputs: list[float] = []
        for x in inputs:
            state, y = self.step(state, x)
            outputs.append(y)
        return outputs


def toy_attention_forward(inputs: Vector) -> list[float]:
    """Toy causal self-attention over sequence to compare computation complexity."""
    seq_len = len(inputs)
    outputs: list[float] = []
    # Attention requires comparing every token against all preceding tokens: O(L^2)
    for i in range(seq_len):
        query = inputs[i]
        scores = [query * inputs[j] for j in range(i + 1)]
        # Softmax
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        weights = [e / sum_exp for e in exp_s]
        # Weighted sum of values
        context = sum(w * inputs[j] for j, w in zip(weights, range(i + 1)))
        outputs.append(context)
    return outputs


def main() -> int:
    state_dim = 4

    print("=" * 68)
    print("状态空间模型 (SSM) 与 Mamba 选择性机制 (Selective Scan) 演示")
    print("=" * 68)

    # 1. Construct a selective copying benchmark sequence
    # Target token at t=1 (value=2.0), followed by 6 filler noise tokens (0.05),
    # query token at t=7 asking to recall the target.
    test_sequence = [0.0, 2.5, 0.05, 0.02, 0.04, 0.01, 0.03, 1.0]
    print(f"输入测试时序 (L={len(test_sequence)}):")
    print(f"  t=1: 关键目标信号 (2.5)")
    print(f"  t=2~6: 无关填充干扰噪音 (~0.03)")
    print(f"  t=7: 读取触发信号 (1.0)\n")

    # 2. Classical LTI SSM evaluation
    classical_model = ClassicalSSM(
        state_dim=state_dim,
        a_diag=[math.exp(rate) for rate in (-0.3, -0.5, -0.8, -1.0)],
        b_weight=[0.5, 0.5, 0.5, 0.5],    # Constant input scale
        c_weight=[0.25, 0.25, 0.25, 0.25],
    )
    classic_outputs = classical_model.forward_sequence(test_sequence)

    print("[1. 经典 LTI 线性时不变 SSM]")
    print("  特点: A, B, C 参数与输入内容无关，具有固定指数衰减率")
    for t, (x, y) in enumerate(zip(test_sequence, classic_outputs)):
        print(f"    t={t}: 输入={x:5.2f} -> 输出={y:6.3f}")
    print(f"  说明: 经过数步衰减和噪音干扰，t=1 的关键信息在 t=7 时已被稀释冲淡\n")

    # 3. Selective SSM (Mamba core idea) evaluation
    selective_model = SelectiveSSM(
        state_dim=state_dim,
        base_a=[-0.1, -0.2, -0.3, -0.4],
    )
    selective_outputs = selective_model.forward_sequence(test_sequence)

    print("[2. 选择性状态空间模型 (Mamba Selective Scan)]")
    print("  特点: Delta/B/C 参数由输入动态生成，遇到干扰噪音主动关小门控 B，保留关键记忆")
    for t, (x, y) in enumerate(zip(test_sequence, selective_outputs)):
        print(f"    t={t}: 输入={x:5.2f} -> 输出={y:6.3f}")
    print("  说明: 手工门控改变输入对状态的贡献；本例未训练，不能据此证明复制任务性能。\n")

    # 4. Attention vs SSM Comparison Table
    print("[3. Transformer 注意力机制 vs Mamba 状态空间模型对比]")
    print(f"  {'维度':<16} | {'Transformer (Self-Attention)':<26} | {'Mamba (Selective SSM)':<24}")
    print("  " + "-" * 72)
    print(f"  {'时间复杂度':<16} | {'O(L^2) 平方增长':<26} | {'O(L) 严格线性':<24}")
    print(f"  {'自回归生成显存':<14} | {'O(L) KV Cache 持续膨胀':<23} | {'O(1) 恒定状态隐空间':<22}")
    print(f"  {'长文本扩展':<16} | {'受限于显存与训练设置':<26} | {'取决于状态容量与任务':<22}")
    print(
        f"  {'硬件加速原理':<14} | {'FlashAttention (SRAM 分块)':<22} | "
        f"{'Hardware-aware Parallel Scan':<20}"
    )
    print("  复杂度比较固定模型维度，仅比较序列混合；不包含训练或 GPU 内核实测。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
