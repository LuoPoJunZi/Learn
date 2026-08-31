# Python 现代 AI 原理案例

这个目录用纯 Python 标准库拆解现代生成式 AI 中常见的基础概念和推理优化方法。案例不会下载模型、不会调用在线 API，也不追求生产性能，重点是让初学者看清数据如何一步一步变化。

## 适合人群

- 已学过变量、列表、循环和函数的 Python 初学者。
- 听说过 Transformer、RAG、Tokenizer 或 LoRA，但不知道它们内部在做什么的人。
- 准备学习 NumPy、PyTorch、Transformers 等工具，希望先建立直觉的人。

## 学习目标

完成本目录后，你应该能够解释：

- 注意力为什么要计算 Query、Key、Value 和 Softmax 权重。
- GQA 如何让多个 Query 头共享更少的 Key/Value 头。
- KV Cache 为什么能减少自回归生成中的重复计算。
- BPE 如何把高频字符组合成子词 Token。
- RAG 为什么要先检索资料，再组织模型输入。
- LoRA 为什么能用两个低秩矩阵减少需要训练的参数。
- 稀疏 MoE 如何按 Token 选择少量专家，而不是激活全部参数。

## 案例索引

| 顺序 | 案例 | 学习重点 | 依赖 |
| :--- | :--- | :--- | :--- |
| 1 | [缩放点积注意力](examples/scaled_dot_product_attention.py) | 相似度、缩放、Softmax、因果遮罩 | 标准库 |
| 2 | [分组查询注意力 GQA](examples/grouped_query_attention.py) | Query 头分组、KV 头共享、缓存量对比 | 标准库 |
| 3 | [KV Cache 自回归解码](examples/kv_cache_decode.py) | 前缀复用、投影次数、输出一致性 | 标准库 |
| 4 | [迷你 BPE Tokenizer](examples/mini_bpe_tokenizer.py) | 子词、词频、合并规则、编码 | 标准库 |
| 5 | [迷你 RAG 检索流程](examples/mini_rag_retrieval.py) | 本地知识库、TF-IDF、余弦相似度、上下文拼装 | 标准库 |
| 6 | [LoRA 低秩更新](examples/lora_low_rank_update.py) | 冻结原权重、低秩矩阵、参数量对比 | 标准库 |
| 7 | [稀疏 MoE 路由](examples/sparse_moe_routing.py) | Router、Top-k 专家、加权输出、稀疏激活 | 标准库 |

## 如何运行

在仓库根目录执行：

```powershell
python "Python\Modern AI\examples\scaled_dot_product_attention.py"
python "Python\Modern AI\examples\grouped_query_attention.py"
python "Python\Modern AI\examples\kv_cache_decode.py"
python "Python\Modern AI\examples\mini_bpe_tokenizer.py"
python "Python\Modern AI\examples\mini_rag_retrieval.py"
python "Python\Modern AI\examples\lora_low_rank_update.py"
python "Python\Modern AI\examples\sparse_moe_routing.py"
```

这些脚本只使用标准库，不需要安装第三方依赖。

## 推荐阅读顺序

1. 先运行每个脚本，不修改代码，观察输入和输出。
2. 阅读 `main()`，先找到示例数据和程序主流程。
3. 再进入各个函数，理解一次计算是怎么完成的。
4. 每次只修改一个变量，例如注意力遮罩、BPE 合并次数、RAG 查询或 LoRA 秩。
5. 最后再使用 NumPy 或 PyTorch 重写矩阵计算，对比代码长度和运行效率。

## 重要边界

- 注意力案例只有单头前向计算，不含训练、位置编码和完整 Transformer 层。
- GQA 案例直接提供 Query/Key/Value 张量，不含真实投影层、批次和 GPU 内核。
- KV Cache 案例只计算一个注意力层的投影次数，不模拟多层缓存、分页、量化或显存搬运。
- BPE 案例按字符教学，生产 Tokenizer 还会处理字节、Unicode、特殊 Token 和批量编码。
- RAG 案例使用 TF-IDF，不含向量数据库、Embedding 模型、重排序器或大语言模型调用。
- LoRA 案例使用预先给定的低秩矩阵，只演示前向更新和参数量，不包含反向传播。
- MoE 案例没有训练 Router、容量限制、负载均衡损失和跨设备专家并行。

这些限制是刻意设计的。先理解最小闭环，再进入大型框架，会比直接面对数千行库代码更容易。

## 推荐修改点

- 把注意力案例的 `causal=True` 改为 `False`，观察每个 Token 能看到哪些位置。
- 把 GQA 的 KV 头从 2 个改成 1 个，观察映射和缓存比例。
- 增加 KV Cache 解码序列长度，比较重复投影次数如何增长。
- 修改 BPE 语料和合并次数，观察词表如何变化。
- 给 RAG 知识库增加一篇文档，再提出能命中它的问题。
- 把 LoRA 的 `rank` 从 1 改为 2，并同步调整矩阵，比较参数量和输出。
- 把 MoE 的 `top_k` 从 2 改成 1，比较激活专家和混合输出。

## 权威参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Hugging Face Transformers: KV Cache Strategies](https://huggingface.co/docs/transformers/kv_cache)
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://papers.nips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9)
- [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

本目录代码是依据论文中的核心思想重新编写的原创教学实现，没有复制论文或外部仓库源码。

## 下一步

- [神经网络标准库示例](<../Neural Network/README.md>)
- [神经网络框架学习路线](<../Neural Network/FRAMEWORK_ROADMAP.md>)
- [Python 环境与依赖管理指南](../ENVIRONMENT_GUIDE.md)
