"""Demonstrate preference alignment from RLHF to DPO."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Vector = list[float]

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferencePair:
    """One human preference: prompt → (preferred, rejected) response."""

    prompt: str
    y_win: str
    y_lose: str


@dataclass
class LogProbTable:
    """Normalized response probabilities derived from trainable log scores.

    表中保存可训练的未归一化分数；查询时对同一 prompt 的候选做 log-softmax。
    概率只覆盖表中的候选，不代表完整语言模型的回答空间。
    """

    table: dict[tuple[str, str], float] = field(default_factory=dict)

    def log_prob(self, prompt: str, response: str) -> float:
        """Return log π(response | prompt)."""
        score = self.table[(prompt, response)]
        scores = [value for (text, _), value in self.table.items() if text == prompt]
        largest = max(scores)
        return score - largest - math.log(sum(math.exp(value - largest) for value in scores))

    def set(self, prompt: str, response: str, value: float) -> None:
        """Assign one finite, unnormalized response log score."""
        if not math.isfinite(value):
            raise ValueError("Response scores must be finite.")
        self.table[(prompt, response)] = value


# ---------------------------------------------------------------------------
# 奖励模型 — RLHF 的第二阶段
# ---------------------------------------------------------------------------


def simple_reward(prompt: str, response: str) -> float:
    """Rate a response with a toy reward function.

    真实系统使用训练好的神经网络；这里用关键词匹配模拟。
    """
    score = 0.0
    positive_keywords = ["安全", "有帮助", "准确", "礼貌"]
    negative_keywords = ["有害", "错误", "冒犯"]
    for word in positive_keywords:
        if word in response:
            score += 1.0
    for word in negative_keywords:
        if word in response:
            score -= 1.5
    return score


# ---------------------------------------------------------------------------
# DPO 数学核心
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Compute σ(x) = 1 / (1 + e^{-x}), numerically stable."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def dpo_loss_single(
    policy: LogProbTable,
    reference: LogProbTable,
    pair: PreferencePair,
    *,
    beta: float,
) -> float:
    """Compute DPO loss for one preference pair.

    L = -log σ( β * [ (log π_θ(y_w|x) - log π_ref(y_w|x))
                     - (log π_θ(y_l|x) - log π_ref(y_l|x)) ] )
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive.")
    log_ratio_win = (
        policy.log_prob(pair.prompt, pair.y_win)
        - reference.log_prob(pair.prompt, pair.y_win)
    )
    log_ratio_lose = (
        policy.log_prob(pair.prompt, pair.y_lose)
        - reference.log_prob(pair.prompt, pair.y_lose)
    )
    margin = beta * (log_ratio_win - log_ratio_lose)
    return max(0.0, -margin) + math.log1p(math.exp(-abs(margin)))


def dpo_loss_batch(
    policy: LogProbTable,
    reference: LogProbTable,
    pairs: list[PreferencePair],
    *,
    beta: float,
) -> float:
    """Average DPO loss over a batch of preference pairs."""
    if not pairs:
        raise ValueError("A preference batch must not be empty.")
    return sum(
        dpo_loss_single(policy, reference, pair, beta=beta) for pair in pairs
    ) / len(pairs)


# ---------------------------------------------------------------------------
# 玩具训练循环 — 用数值梯度更新策略
# ---------------------------------------------------------------------------


def numerical_gradient(
    policy: LogProbTable,
    reference: LogProbTable,
    pairs: list[PreferencePair],
    *,
    beta: float,
    epsilon: float = 1e-4,
) -> dict[tuple[str, str], float]:
    """Estimate ∂L/∂θ for every entry in the policy table."""
    grads: dict[tuple[str, str], float] = {}
    for key in policy.table:
        original = policy.table[key]
        policy.table[key] = original + epsilon
        loss_plus = dpo_loss_batch(policy, reference, pairs, beta=beta)
        policy.table[key] = original - epsilon
        loss_minus = dpo_loss_batch(policy, reference, pairs, beta=beta)
        policy.table[key] = original
        grads[key] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return grads


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    # ── 1. RLHF 三阶段流水线概览 ──────────────────────────────
    print("=" * 60)
    print("[阶段] RLHF 三阶段流水线 vs DPO 直接优化")
    print("=" * 60)
    print("""
  RLHF 流水线:
    阶段1  SFT        → 用高质量数据微调基础模型
    阶段2  奖励模型    → 用人类偏好训练一个打分网络
    阶段3  PPO        → 用强化学习让策略最大化奖励

  DPO 直接优化:
    阶段1  SFT        → 同上
    阶段2  DPO        → 跳过奖励模型，直接用偏好对优化策略
    优势: 无需训练奖励模型，无需复杂的 PPO 超参数调节
""")

    # ── 2. 构造偏好数据集 ──────────────────────────────────────
    pairs = [
        PreferencePair("如何学编程？", "制定计划，从基础语法开始，安全有帮助", "随便看看，错误也没关系"),
        PreferencePair("Python好学吗？", "Python 语法简洁，准确来说很适合初学者", "不好学，冒犯到你了吧"),
        PreferencePair("推荐一本书？", "《流畅的Python》，内容准确且有帮助", "去网上随便搜，有害建议也行"),
    ]

    print("[偏好对] 人类标注的偏好数据:")
    for i, pair in enumerate(pairs, 1):
        r_win = simple_reward(pair.prompt, pair.y_win)
        r_lose = simple_reward(pair.prompt, pair.y_lose)
        print(f"  对 {i}: prompt=\"{pair.prompt}\"")
        print(f"    ✓ 优选: \"{pair.y_win}\"  奖励={r_win:+.1f}")
        print(f"    ✗ 劣选: \"{pair.y_lose}\"  奖励={r_lose:+.1f}")
    print()

    # ── 3. 初始化策略和参考模型 ────────────────────────────────
    reference = LogProbTable()
    policy = LogProbTable()
    for pair in pairs:
        # 参考模型：对两个回答概率差不多（SFT 后的初始状态）
        reference.set(pair.prompt, pair.y_win, -1.0)
        reference.set(pair.prompt, pair.y_lose, -1.2)
        # 策略模型：从和参考模型相同的值开始
        policy.set(pair.prompt, pair.y_win, -1.0)
        policy.set(pair.prompt, pair.y_lose, -1.2)

    beta = 0.1
    learning_rate = 0.5
    steps = 8

    # ── 4. DPO 训练循环 ───────────────────────────────────────
    print("[损失] DPO 训练过程 (β={:.1f}, lr={:.1f}):".format(beta, learning_rate))
    for step in range(steps):
        loss = dpo_loss_batch(policy, reference, pairs, beta=beta)
        grads = numerical_gradient(policy, reference, pairs, beta=beta)

        for key in policy.table:
            policy.table[key] -= learning_rate * grads[key]

        if step % 2 == 0 or step == steps - 1:
            print(f"  步骤 {step}: 平均损失 = {loss:.4f}")
    print()

    # ── 5. 训练结果 ───────────────────────────────────────────
    print("[结果] 策略模型概率变化（训练后 vs 参考模型）:")
    for pair in pairs:
        ref_ratio = sigmoid(
            reference.log_prob(pair.prompt, pair.y_win)
            - reference.log_prob(pair.prompt, pair.y_lose)
        )
        pol_ratio = sigmoid(
            policy.log_prob(pair.prompt, pair.y_win)
            - policy.log_prob(pair.prompt, pair.y_lose)
        )

        print(f"  prompt: \"{pair.prompt}\"")
        print(f"    参考模型偏好优选概率: {ref_ratio:.1%}")
        print(f"    策略模型偏好优选概率: {pol_ratio:.1%} {'↑ 对齐成功' if pol_ratio > ref_ratio else ''}")

    # ── 6. 核心数学对比 ───────────────────────────────────────
    print()
    print("=" * 60)
    print("[对比] 为什么 DPO 能省掉奖励模型？")
    print("=" * 60)
    print("""
  RLHF 目标:  max_θ E[ R(x,y) - β·KL(π_θ ∥ π_ref) ]
    → 需要先训练 R(x,y)，再用 PPO 优化策略

  DPO 关键洞察:  最优策略的闭式解为
    π*(y|x) = π_ref(y|x) · exp( R(x,y) / β ) / Z(x)
    → 反推出隐式奖励: R(x,y) = β · log(π_θ(y|x) / π_ref(y|x)) + β·log Z
    → 用概率比表示奖励，比较同一 prompt 的回答时，归一化项 log Z 相消
    → DPO 损失直接用策略概率比表达，无需单独的奖励模型
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
