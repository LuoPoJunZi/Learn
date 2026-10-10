# RoPE 与分块注意力：从位置到内存

## 适合人群与学习目标

适合已经理解列表、循环、函数和点积注意力的读者。先读 [本目录索引](README.md)，运行 [基础注意力](examples/scaled_dot_product_attention.py)，再阅读本页。

本页帮助你回答两个问题：注意力如何区分词语的位置？为什么不保存完整注意力矩阵也能计算同一个结果？

| 主题 | 本地代码 | 运行环境 | 观察重点 |
| :--- | :--- | :--- | :--- |
| 旋转位置编码 RoPE | [rotary_position_embedding.py](examples/rotary_position_embedding.py) | Python 3.10+，标准库 | 向量长度、相对位置、整体平移 |
| 在线 Softmax 分块注意力 | [online_softmax_attention.py](examples/online_softmax_attention.py) | Python 3.10+，标准库 | 分块累积、稳定性、与完整计算一致 |

两个文件只打印结果，不写文件，不下载模型。

## 一、RoPE 给位置一个旋转角度

### 为什么还需要位置信息

普通点积只比较向量内容。如果没有任何位置机制，模型很难仅凭这个运算分辨“我喜欢你”和“你喜欢我”的词序。RoPE 把位置写进 Query 和 Key 的旋转角度，使注意力得分同时包含内容和相对位置关系。

本例把四维向量分成相邻坐标对：`[x0, x1]` 和 `[x2, x3]`。对于每一对，做二维旋转：

```text
frequency_i = base^(-2*i/d)
angle = position * frequency_i

new_left  = left * cos(angle) - right * sin(angle)
new_right = left * sin(angle) + right * cos(angle)
```

这里 `i` 从 0 开始，`d` 是向量维度。不同坐标对使用不同频率，所以长向量可以同时表达多种位置变化尺度。代码中的 `start` 等于 `2*i`。

### 如何运行和阅读输出

在仓库根目录运行下面的命令，它会检查三个旋转性质并打印不同相对距离的得分：

```powershell
python "Python\Modern AI\examples\rotary_position_embedding.py"
```

本例位置是 `(2, 5)`，两个位置同时加 10 后，得分依然约为 `-1.357341`。程序还计算 `query` 与“只旋转相对距离 3 的 key”的点积，结果相同。

这对应下面的恒等式：

```text
dot(R_m(q), R_n(k)) = dot(q, R_(n-m)(k))
```

旋转也保持向量长度，程序检查平方长度误差接近零。阅读顺序是 `main()` 中的三个比较，再看 `rotary_embedding()` 的坐标对循环。

### 典型场景和常见误解

- 场景：理解 Transformer 位置编码，并检查带 KV Cache 的解码位置是否连续。
- 相对得分不一定随距离严格递减。本例得分会振荡；不要把旋转解释成固定的距离惩罚。
- 位置平移恒等式要求比较的是相同的原始 q/k 与相同频率，不能据此断言完整语言模型在任意平移下输出都不变。
- 本例采用相邻坐标配对。其他实现可能使用前后半向量配对；频率和权重布局必须匹配，不能混用。
- 改 `base` 不是完整的长上下文扩展方案；本例不实现训练、RoPE scaling 或上下文长度验证。

## 二、分块注意力让历史贡献正确累积

### 先看完整算法

对于一个 Query，先算所有 Key 的得分，然后 Softmax，最后对 Value 加权求和。参考实现 `dense_attention()` 会保存 `N x N` 得分矩阵。

如果只读取一小块 Key/Value，直接给每一块单独做 Softmax 再平均会出错：每一块的分母不同，失去了整个序列的权重比例。

### 需要保留的三个状态

`online_attention()` 对每个 Query 保存：

- `largest`：目前见过的最大得分。
- `denominator`：相对当前最大值计算的指数和。
- `numerator`：同一尺度下的 Value 加权和。

新块到来时，最大值可能变大，旧状态必须同步缩放：

```text
new_max = max(old_max, max(block_scores))
correction = exp(old_max - new_max)
block_weights = exp(block_scores - new_max)

new_denominator = correction * old_denominator + sum(block_weights)
new_numerator = correction * old_numerator + sum(block_weights * block_values)
output = new_numerator / new_denominator
```

第一块之前 `old_max=-inf`，因此旧贡献的缩放因子是 0。以后每次改变最大值，都同时修正分子和分母，才能维持同一个 Softmax 结果。

### 运行与结果

下面的命令比较块大小为 1、3、8、16 时的输出，并分别检查因果和双向注意力：

```powershell
python "Python\Modern AI\examples\online_softmax_attention.py"
```

8 个 Token 的完整参考矩阵有 64 个得分。在线计算每次只保存一个得分块和累积状态；Q/K/V 输入和输出仍然需要内存。本次验证的最大差异约为 `1e-16`，属于浮点计算顺序带来的误差。

程序还使用约 `1e6` 的大得分检验稳定性。减去最大值后再取指数，可以避免直接 `exp(1000000)` 的溢出。

### 它与 FlashAttention 的关系

在线 Softmax 和分块是理解 FlashAttention 的重要基础。真正的 FlashAttention 还需要 GPU 内核、存储层次、线程调度和反向传播设计；本例逐个 Query 使用 Python 循环。

本例依然进行全部可见的点积计算，双向注意力计算量仍是 `O(N^2*d)`。它展示的是临时得分存储的变化，不是减少为线性计算量，也没有 GPU 加速测量。

本例要求 Q/K/V 的 Token 数量一致，不支持跨注意力、增量解码的不同序列长度或 padding 遮罩。因果模式总会保留当前位置，所以不会出现整行都被遮住的情况。

## 少量修改与理解检查

1. 把 RoPE 的两个位置同时增加相同整数。参考结果：相同原始 q/k 的点积近似不变。
2. 把分块注意力的块大小改为 2 或 5。参考结果：只改变计算分块方式，结果仍应接近完整参考值。
3. 删除分子中的 `correction`。参考结果：后续块提高最大值时，旧 Value 贡献过大，比较检查可能失败。

## 来源与使用提醒

资料核对日期：2026-10-10。本页和源码按公式原创编写，未复制外部仓库代码。

- [RoFormer 论文](https://arxiv.org/abs/2104.09864)：位置旋转与相对位置关系。
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)：精确注意力中的分块与 GPU IO 思路。
- [PyTorch SDPA 官方教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)：把公式学习衔接到框架调用；具体后端由输入与硬件条件决定。

本地代码遵循仓库 [LICENSE](../../LICENSE)。外部论文、框架文档与源码按各自许可使用，引用不等于可以整段复制。

## 下一步入口

- [GQA 与 KV 头共享](examples/grouped_query_attention.py)
- [KV Cache 解码](examples/kv_cache_decode.py)
- [神经网络框架学习路线](<../Neural Network/FRAMEWORK_ROADMAP.md>)
