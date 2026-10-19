# WebGPU 初学导引与通用计算实验台

这个案例展示了下一代 Web 标准 **WebGPU** 的基础架构及其在浏览器中执行纯数学并行通用计算（GPGPU）的核心流程。

## 适合人群

- 了解基础 JavaScript，希望探索硬件加速与底层渲染/计算原理的初学者。
- 听说过端侧浏览器大模型推理（如 Transformers.js、WebLLM），好奇“浏览器不依赖后端是如何跑 AI”的开发者。
- 了解 WebGL 但对现代现代显卡 API（DirectX 12、Vulkan、Metal）设计思想感兴趣的学习者。

## 学习目标

完成本案例后，你应该能够：

- 解释为什么 WebGPU 相比古老的 WebGL 能显著释放现代多核 GPU 的算力。
- 掌握 WebGPU 计算管线（Compute Pipeline）的核心 5 步流程：`Adapter` -> `Device` -> `Buffer` -> `WGSL Shader` -> `Dispatch`。
- 读懂基础 WGSL（WebGPU Shading Language）着色器语法。
- 掌握在前端进行 WebGPU 特性检测与平稳降级的实践方法。

## 文件说明

| 文件 | 职责 |
| :--- | :--- |
| [index.html](index.html) | 单文件演示页面，包含 WebGPU 环境检测、WGSL 着色器构建与向量并行相加演示 |

## 如何打开

可以先直接打开 [index.html](index.html)。如果浏览器的文件协议策略不可用，在仓库根目录启动一个仅监听本机的静态服务器：

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory HTML/Examples/webgpu-basics
```

浏览器访问 `http://127.0.0.1:8765/`；结束后在终端按 `Ctrl+C` 停止。该命令只提供静态文件。

**浏览器环境支持要求**：
WebGPU 需要安全上下文（例如 HTTPS 或可信 localhost），并受浏览器、系统、显卡驱动和硬件加速设置共同影响。`navigator.gpu` 存在也不保证能申请到适配器或设备，应以实际请求结果为准；各浏览器版本范围见下方 API 参考的兼容表。

若当前环境不支持硬件加速，页面顶部会呈现友好的降级警告提示。

## 观察顺序

1. **观察特性检测状态条**：页面加载时自动执行 `navigator.gpu.requestAdapter()`，识别本地物理独立或集成显卡。
2. **阅读 WGSL 源码结构**：查看代码中如何使用 `@compute @workgroup_size(64)` 声明计算线程网格，以及如何通过 `@binding` 绑定内存缓冲区。
3. **点击“在 GPU 上启动计算”**：观察 JavaScript 如何建立 GPU 显存 Buffer、写入数据、录制命令（Command Encoder）并派发执行，最后通过 `mapAsync` 取回计算结果。

## 推荐修改点

- 尝试修改 JavaScript 中的两组输入向量为相同长度（例如 1024），观察结果。工作组数量按 `Math.ceil(length / 64)` 派发，并在着色器中检查末尾越界。
- 尝试修改 WGSL 着色器中的运算逻辑（例如将加法改为乘法 `inputA[i] * inputB[i]`）。

每次运行结束会释放 Buffer 与 Device，计算期间禁用按钮。输入必须是同长度、非空、数值有限的 `Float32Array`；本例不以四元素演示宣称 GPU 比 CPU 更快。

## 权威参考

- [W3C WebGPU 规范草案](https://www.w3.org/TR/webgpu/)
- [W3C WebGPU Shading Language (WGSL) 规范](https://www.w3.org/TR/wgsl/)
- [MDN: WebGPU API 指南与参考](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)

## 下一步

- [现代 Web API 实验台](../modern-web-apis/README.md)
- [现代 CSS 新特性实验台](../modern-css-features/README.md)
- [现代 JavaScript ES2024-2026 新特性](../es2024-2026-features/README.md)
