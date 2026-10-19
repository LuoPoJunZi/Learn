"""Demonstrate vision-language model (VLM) patch projection and unified multimodal attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class VisionPatch:
    """An extracted 2D spatial image patch."""

    patch_id: int
    row: int
    col: int
    pixels: Vector  # Flattened pixel intensity values


@dataclass(frozen=True)
class MultimodalToken:
    """A token in the unified sequence, either visual or textual."""

    token_id: int
    modality: str  # 'vision' or 'text'
    label: str
    embedding: Vector


def extract_patches_from_image(
    image: Matrix,
    patch_size: int = 2,
) -> list[VisionPatch]:
    """Slice an HxW 2D image into non-overlapping patches of size patch_size."""
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")
    if not image or not image[0] or any(len(row) != len(image[0]) for row in image):
        raise ValueError("The image must be non-empty and rectangular.")
    h = len(image)
    w = len(image[0])
    if h % patch_size or w % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size in this example.")
    patches: list[VisionPatch] = []
    pid = 0

    for r in range(0, h, patch_size):
        for c in range(0, w, patch_size):
            pixels: list[float] = []
            for dr in range(patch_size):
                for dc in range(patch_size):
                    pixels.append(image[r + dr][c + dc])
            patches.append(VisionPatch(
                patch_id=pid, row=r // patch_size, col=c // patch_size, pixels=pixels,
            ))
            pid += 1

    return patches


def project_vision_to_llm_dim(
    patch_pixels: Vector,
    projection_weights: Matrix,
    projection_bias: Vector,
) -> Vector:
    """Linear projector (like in LLaVA) mapping vision space to LLM hidden space."""
    if not patch_pixels or not projection_weights:
        raise ValueError("Patch values and projection weights must be non-empty.")
    if len(projection_weights) != len(projection_bias):
        raise ValueError("Projection rows and biases must have matching dimensions.")
    if any(len(row) != len(patch_pixels) for row in projection_weights):
        raise ValueError("Each projection row must match the flattened patch dimension.")
    return [
        sum(p * w for p, w in zip(patch_pixels, row)) + b
        for row, b in zip(projection_weights, projection_bias)
    ]


def dot_product(a: Vector, b: Vector) -> float:
    """Compute dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError("Embedding dimensions must match.")
    return sum(x * y for x, y in zip(a, b))


def softmax(values: Vector) -> Vector:
    """Stable softmax function."""
    max_val = max(values)
    exp_vals = [math.exp(v - max_val) for v in values]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]


def compute_cross_modal_attention(
    tokens: list[MultimodalToken],
) -> Matrix:
    """Calculate scaled dot-product attention scores across all multimodal tokens."""
    if not tokens or not tokens[0].embedding:
        raise ValueError("At least one non-empty embedding is required.")
    d_model = len(tokens[0].embedding)
    scale = 1.0 / math.sqrt(d_model)
    attention_matrix: Matrix = []

    for q_tok in tokens:
        raw_scores = [
            dot_product(q_tok.embedding, k_tok.embedding) * scale
            for k_tok in tokens
        ]
        weights = softmax(raw_scores)
        attention_matrix.append(weights)

    return attention_matrix


def main() -> int:
    print("=" * 68)
    print("视觉多模态大模型 (VLM) 架构与跨模态注意力演示")
    print("=" * 68)

    # 1. Simulate a 4x4 grayscale toy image
    # Top-half: bright sky/light (values ~ 0.8), bottom-half: dark ground (values ~ 0.2)
    # Bottom-right: a bright object (1.0)
    image: Matrix = [
        [0.85, 0.90, 0.88, 0.82],
        [0.80, 0.85, 0.86, 0.80],
        [0.20, 0.22, 0.30, 0.95],
        [0.18, 0.25, 0.92, 1.00],
    ]
    patch_size = 2
    patches = extract_patches_from_image(image, patch_size=patch_size)

    print(f"[1. 图像切片与展平 (Patch Slicing, {len(image)}x{len(image[0])} -> {len(patches)} Patches)]")
    for p in patches:
        print(f"  Patch #{p.patch_id} (位置 [{p.row},{p.col}]): 像素点={p.pixels}")

    # 2. Modality Projector (maps 4-dim patch pixels to 3-dim LLM space)
    proj_weights = [
        [0.5, 0.5, 0.1, 0.1],
        [0.1, 0.2, 0.7, 0.7],
        [0.3, 0.3, 0.3, 0.3],
    ]
    proj_bias = [0.0, 0.1, -0.05]

    vision_tokens: list[MultimodalToken] = []
    for p in patches:
        llm_rep = project_vision_to_llm_dim(p.pixels, proj_weights, proj_bias)
        vision_tokens.append(
            MultimodalToken(
                token_id=p.patch_id,
                modality="vision",
                label=f"IMG_P{p.patch_id}",
                embedding=[round(v, 3) for v in llm_rep],
            )
        )

    print("\n[2. 手工模态投影器转换 (Patch Pixels -> Hidden Dim = 3)]")
    for vt in vision_tokens:
        print(f"  {vt.label} ({vt.modality}): 映射向量 = {vt.embedding}")

    # 3. Interleaving Text Tokens
    # User query: "图中 右下角 是什么？"
    text_tokens: list[MultimodalToken] = [
        MultimodalToken(4, "text", "这幅图", [0.4, 0.1, 0.3]),
        # Handcrafted embedding; no learned spatial alignment is demonstrated.
        MultimodalToken(5, "text", "右下角", [0.1, 0.85, 0.25]),
        MultimodalToken(6, "text", "物体", [0.2, 0.75, 0.35]),
    ]

    unified_sequence = vision_tokens + text_tokens
    print(f"\n[3. 统一多模态 Token 序列拼装 (Total={len(unified_sequence)} Tokens)]")
    seq_repr = " ".join(f"[{t.label}]" for t in unified_sequence)
    print(f"  输入流: {seq_repr}\n")

    # 4. Multimodal Attention Computation
    attn_matrix = compute_cross_modal_attention(unified_sequence)

    print("[4. 统一注意力权重矩阵 (文本词对图像 Patch 的注意力分布)]")
    target_idx = 5  # "右下角" text token
    query_tok = unified_sequence[target_idx]
    weights_for_query = attn_matrix[target_idx]

    print(f"  查询词: '{query_tok.label}'")
    for i, tok in enumerate(unified_sequence):
        w = weights_for_query[i]
        bar = "█" * int(w * 30)
        print(f"    -> 关注 [{tok.label:<7}]: 权重={w:5.3f} | {bar}")

    print("\n[5. 核心原理与动态分辨率 (Dynamic Resolution)]")
    print("  - 传统方案: 固定粗糙缩放图片至 224x224 或 336x336，造成细小文字/图表 OCR 模糊失真")
    print(
        "  - Qwen2-VL / LLaVA-NeXT 使用动态分辨率等机制；"
        "本例只有固定切片，没有实现这些完整架构。"
    )
    print("  - 本例使用手工向量，无视觉编码器训练、位置编码、Value 加权或生成能力。")
    print("  - 矩阵仅演示混合模态的点积权重；不能据此证明学会了物体识别或空间定位。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
