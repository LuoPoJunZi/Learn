# 多头潜在注意力 MLA 原理入门：把 KV Cache 压缩进低维潜在空间

## 适合人群与学习目标

适合已经读过 [缩放点积注意力](examples/scaled_dot_product_attention.py)、[分组查询注意力 GQA](examples/grouped_query_attention.py) 和 [KV Cache 自回归解码](examples/kv_cache_decode.py) 的初学者。

资料核验：**2026-11-20**，核心原理参考 DeepSeek-V2 论文中的 Multi-Head Latent Attention（MLA）设计。完成后你应该能够：

- 解释为什么超长上下文（几十万 Token）下，KV Cache 会成为推理显存的主要瓶颈。
- 对比 MHA、GQA 与 MLA 在“缓存什么”上的根本差异。
- 看懂 MLA 的两步流程：**下投影压缩入缓存** 与 **上投影按头恢复**。
- 运行本仓库的纯标准库示例并验证缓存压缩倍率与边界错误处理。

---

## 为什么有了 GQA 还需要 MLA？

自回归生成时，模型每生成一个新 Token，都要把之前所有 Token 的 Key 和 Value 留在显存里（即 KV Cache）：

```text
MHA（多头注意力） : 每个 Query 头都配独立的 K 头和 V 头  -> 表达能力强，但 KV Cache 最大
GQA（分组查询）   : 多个 Query 头共享同一组 K/V 头        -> 显存下降，但头数压得太少会损精度
MLA（多头潜在）   : 不直接存各头 K/V，只存低维潜在向量 c_t -> 压缩率极高，同时保留多头表达能力
```

以单层 32 个头、每头维度 `head_dim = 128` 为例，单个 Token 需要缓存的浮点数个数对比：

| 机制 | 单 Token 缓存公式 | 单 Token 标量数 | 相对 MHA 压缩比 |
| :--- | :--- | :--- | :--- |
| **MHA** | $2 \times n_h \times d_h$ | $2 \times 32 \times 128 = 8192$ | `1.0x` |
| **GQA（8 组 KV 头）** | $2 \times n_{kv} \times d_h$ | $2 \times 8 \times 128 = 2048$ | `4.0x` |
| **MLA（潜在维 512）** | $d_c$ | `512` | **`16.0x`** |

---

## MLA 的核心两步计算

给定第 $t$ 个 Token 的隐藏向量 $x_t \in \mathbb{R}^{d_{\text{model}}}$：

1. **下投影压缩（写入 KV Cache）**
   $$c_t^{KV} = W_{DKV} x_t \in \mathbb{R}^{d_c}$$
   推理时缓存里**只保存**低维向量 $c_t^{KV}$，不再保存展开后的 $K$ 和 $V$。

2. **上投影恢复（按头计算注意力）**
   对第 $h$ 个注意力头，用该头专属的上投影矩阵恢复键和值：
   $$k_t^{(h)} = W_{UK}^{(h)} c_t^{KV}, \qquad v_t^{(h)} = W_{UV}^{(h)} c_t^{KV}$$
   再用常规缩放点积注意力和因果遮罩计算加权上下文输出。

> [!NOTE]
> **进阶工程技巧（矩阵吸收）**：在真实推理框架中，因为 $q_t^T k_j = (W_Q x_t)^T (W_{UK} c_j^{KV}) = x_t^T (W_Q^T W_{UK}) c_j^{KV}$，可以把 $W_{UK}$ 提前“吸收”进 $W_Q$，从而在计算点积时甚至不需要显式把 $c_j^{KV}$ 展开成 $k_j$。本仓库教学代码为了让你一眼看清数据形状，保留了显式上投影步骤。

---

## 运行与验证

示例要求 Python 3.10+，仅使用标准库，不下载模型、不联网：

```powershell
python "Python\Modern AI\examples\multi_head_latent_attention.py"
python -m unittest discover -s .github/scripts/tests -p "test_frontier_ai_2026.py" -v
```

运行后会打印：
1. 每个 Token 压缩后的 2 维潜在缓存向量；
2. 第 0 个头的因果注意力权重矩阵（严格下三角，未来位置权重为 `0.0`）；
3. 拼接后的多头输出；
4. 教学尺寸与真实尺寸（4096 序列、32 头）下 MHA / GQA / MLA 的 KV Cache 标量数对比。

---

## 常见误区与重要边界

- **误区 1：MLA 会让计算变慢很多吗？** 压缩和恢复确实增加了矩阵乘法，但在长上下文解码阶段，GPU 往往受限于**显存带宽**（Memory-bound）而非算力，大幅减少 KV Cache 读写反而显著提升吞吐。
- **误区 2：RoPE 位置编码可以直接套在压缩后的 $c_t^{KV}$ 上吗？** 不行。因为旋转位置编码是非线性的（依赖位置），会破坏矩阵吸收。真实 DeepSeek-V2/V3 采用的是 **Decoupled RoPE**（把一小段专门带位置编码的 Key 与压缩潜在向量拼在一起）。
- **实现边界**：[multi_head_latent_attention.py](examples/multi_head_latent_attention.py) 演示压缩、恢复和缓存对比的核心数学流程，不包含 Decoupled RoPE、Query 低秩压缩、输出投影 $W_O$、反向传播训练或 GPU 内核。

---

## 权威参考与下一步

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)（MLA 原始论文）
- [分组查询注意力 GQA 示例](examples/grouped_query_attention.py)
- [RoPE 与分块注意力中文讲解](ATTENTION_MECHANICS_GUIDE.md)
- [组相对策略优化 GRPO 入门](GRPO_GUIDE.md)
