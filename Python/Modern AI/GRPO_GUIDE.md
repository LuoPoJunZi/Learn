# 组相对策略优化 GRPO 入门：去掉 Critic 模型的推理强化学习

## 适合人群与学习目标

适合已经读过 [RLHF 到 DPO 对齐原理](examples/rlhf_to_dpo_alignment.py) 和 [测试时计算扩展与慢思考](examples/test_time_compute_scaling.py) 的初学者。

资料核验：**2026-11-20**，原理参考 DeepSeekMath 与 DeepSeek-R1 论文中提出的 **GRPO（Group Relative Policy Optimization）**。完成后你应该能够：

- 对比 PPO、DPO 与 GRPO 三种后训练方法的核心结构差异。
- 理解 GRPO 为什么能去掉显存巨大的独立 Critic（价值模型）。
- 看懂组内相对优势（Group Relative Advantage）计算、零方差边界以及非负 KL 惩罚项。
- 运行纯标准库示例并观察高奖励推理路径的概率如何提升。

---

## PPO、DPO 与 GRPO 有什么区别？

| 算法 | 需要独立奖励模型？ | 需要独立 Critic 价值网络？ | 核心优化思路 | 典型场景 |
| :--- | :--- | :--- | :--- | :--- |
| **PPO (RLHF)** | 需要 | **需要**（通常和策略模型一样大） | 用 Critic 估计基线，在线采样更新 | 通用对话对齐 |
| **DPO** | 不需要 | 不需要 | 在离线偏好对 $(y_{\text{win}}, y_{\text{lose}})$ 上直接拉大概率比 | 离线偏好微调 |
| **GRPO** | 可选（常用规则/单测验证器） | **不需要** | 对同一问题采样一组 $G$ 个回答，用**组内均值和标准差**当基线 | 数学、代码等可验证推理 |

```text
PPO  : Prompt -> 1 个回答 -> Critic 网络估算基线 -> 计算优势 A
GRPO : Prompt -> 一组 G 个回答 -> 规则验证器打分 [r_1..r_G] -> 组内归一化算相对优势 A_i
```

---

## 核心公式拆解

### 1. 组内相对优势（无需 Critic）

对同一个问题采样一组 $G$ 个回答，拿到奖励列表 $\mathbf{r} = [r_1, r_2, \dots, r_G]$：

$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \varepsilon}$$

- 得分高于组平均的回答，$A_i > 0$（鼓励提高生成概率）；
- 得分低于组平均的回答，$A_i < 0$（压低生成概率）；
- **零方差边界**：如果组里所有回答全对或全错（$\text{std}(\mathbf{r}) = 0$），本例显式返回全 `0.0`，避免除零或产生无效梯度。

### 2. 裁剪代理目标 + 非负 KL 惩罚

为防止单步更新跨度过大（策略崩塌）或偏离参考模型太远，GRPO 结合了概率比裁剪和 Schulman 非负 KL 估计：

$$\rho_i = \frac{\pi_\theta(o_i)}{\pi_{\text{old}}(o_i)}, \qquad D_{KL}(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{\pi_{\text{ref}}(o_i)}{\pi_\theta(o_i)} - \log\frac{\pi_{\text{ref}}(o_i)}{\pi_\theta(o_i)} - 1$$

每个样本的目标项为：

$$\min\Big(\rho_i A_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\Big) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

---

## 运行与验证

示例要求 Python 3.10+，仅使用标准库，完全离线：

```powershell
python "Python\Modern AI\examples\grpo_group_relative_policy.py"
python -m unittest discover -s scripts/tests -p "test_frontier_ai_2026.py" -v
```

运行后会展示：
1. 奖励 `[1.0, 0.2, 0.0, 1.0]` 如何转化为正负相对优势；
2. 全相同奖励 `[1.0, 1.0, 1.0]` 时如何安全返回全 `0.0`；
3. 经过 10 步玩具梯度上升后，两个高分回答的概率显著上升，错误回答的概率显著下降。

---

## 常见误区与重要边界

- **误区 1：GRPO 一定比 DPO 好吗？** 不是。GRPO 特别适合“有明确验证器”的任务（数学答案核对、代码跑单元测试）；对于开放式写作风格偏好，离线偏好对的 DPO 往往更简单经济。
- **误区 2：组大小 $G$ 越小越省算力，设成 2 可以吗？** $G$ 太小时组内均值和方差噪声极大，还容易经常出现全对或全错（零优势浪费采样）。工程中通常每题采样 $8 \sim 64$ 个样本。
- **实现边界**：[grpo_group_relative_policy.py](examples/grpo_group_relative_policy.py) 在 4 个离散候选回答的玩具概率表上演示目标函数和数值梯度更新，不包含 Token 级自回归展开、真实神经网络权重或分布式 RL 训练框架。

---

## 权威参考与下一步

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)（GRPO 原始提出）
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Approximating KL Divergence (John Schulman)](http://joschu.net/blog/kl-approx.html)
- [RLHF 到 DPO 对齐示例](examples/rlhf_to_dpo_alignment.py)
- [多头潜在注意力 MLA 原理入门](MLA_GUIDE.md)
