# MATLAB 与 PyTorch / ONNX 模型互通

在当今的 AI 与工程研发中，团队往往面临技术栈分工的现实：算法研究团队习惯使用 Python、PyTorch 或 Hugging Face 快速训练最新的深度学习模型；而系统仿真、控制工程与嵌入式部署团队则重度依赖 MATLAB 和 Simulink 进行严格的物理系统在环（HIL）验证。

过去，这种生态壁垒导致算法交付困难重重。借助 **ONNX（Open Neural Network Exchange，开放神经网络交换格式）** 和现代 MATLAB 的导入导出套件，开发者可以轻松实现双向模型无缝流转：既能将外界训练好的 PyTorch 大模型导入 MATLAB 进行控制仿真，也能将 MATLAB 训练的专用模型导出部署至边缘设备。

---

## 适合人群

- 使用 Python 训练了深度学习网络，需要将模型放入 Simulink 进行闭环仿真的控制工程师。
- 团队中有跨语言协作需求（Python 研究员 + MATLAB 工程师）的开发者。
- 想在 MATLAB 内部直接加载外部开源神经网络进行图像、语音或时序推断的学习者。

## 学习目标

完成本章后，你应该能够：

- 理解 ONNX 标准协议在跨框架（PyTorch ↔ MATLAB）协作中的通用桥梁作用。
- 使用 `importONNXNetwork` 或 `importNetworkFromPyTorch` 直接将外部模型导入为现代 `dlnetwork`。
- 掌握计算机视觉与时序信号在 Python 与 MATLAB 间的**核心张量维度排列差异（NCHW vs SSCB）**。
- 使用 `Deep Network Designer` 可视化检查网络层级连接、输入通道与权重形状。
- 将模型封装为 Simulink 内部的 Neural Network Block，完成“AI 代理模型 + 物理动力学”联合仿真。

---

> [!IMPORTANT]
> **双端前置依赖自查：**
> 1. **Python 端**：导出 ONNX 需在 Python 环境中安装 PyTorch 与 ONNX 库：
>    ```bash
>    python -m pip install torch torchvision onnx onnxscript
>    ```
> 2. **MATLAB 端**：调用 `importONNXNetwork` 与 `exportONNXNetwork` 必须安装 **Deep Learning Toolbox** 以及官方免费扩展包 **Deep Learning Toolbox Converter for ONNX Model Format**（在 MATLAB 顶部【主页】->【附加功能】中搜索安装）。

---

## 生态互联流转架构

```text
[PyTorch / TensorFlow / JAX]
        │
        ▼ (torch.onnx.export 或 直接保存权重)
   [标准 ONNX 模型文件 (.onnx)]
        │
        ▼ (importONNXNetwork)
[MATLAB 工作区 (dlnetwork / DAGNetwork)]
   ├── 1. Deep Network Designer 交互式可视化分析
   ├── 2. 现代批量推理: minibatchpredict / dlfeval
   ├── 3. 嵌入 Simulink 模块: 充当非线性控制器或物理系统代理模型
   └── 4. MATLAB Coder / GPU Coder: 一键生成纯 C++/CUDA 嵌入式部署源码
```

---

## 第一步：从 PyTorch 导出为 ONNX 标准格式 (Python 端)

在 Python 环境中，通过几行代码将一个训练好的 PyTorch 分类模型导出为 `.onnx` 格式：

```python
# python_export.py (在 Python 环境运行)
import torch
import torchvision.models as models

# 1. 实例化模型（此处以轻量 ResNet-18 为例）
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
# DEFAULT 首次使用会下载权重到 PyTorch 缓存；导出在单独的实验目录进行。

# 2. 构造一个虚拟输入张量，声明 batch 大小与输入分辨率
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 导出为 ONNX 模型
torch.onnx.export(
    model,
    dummy_input,
    "resnet18_model.onnx",
    input_names=["input_tensor"],
    output_names=["class_logits"],
    opset_version=17,
    dynamo=True,
)
print("ONNX 模型导出成功: resnet18_model.onnx")
```

---

## 第二步：在 MATLAB 中导入与可视化拓扑

将生成的 `resnet18_model.onnx` 放入 MATLAB 当前工作目录中。

在 MATLAB 命令行执行：

```matlab
% 1. 新版优先使用 importNetworkFromONNX；较旧版本保留原导入入口
if exist('importNetworkFromONNX', 'file')
    net = importNetworkFromONNX('resnet18_model.onnx', ...
        'InputDataFormats', 'BCSS', 'OutputDataFormats', 'BC');
else
    net = importONNXNetwork('resnet18_model.onnx', 'TargetNetwork', 'dlnetwork');
end

% 2. 检查网络输入与输出层签名
disp('网络输入层属性:');
disp(net.InputNames);
disp('网络输出层属性:');
disp(net.OutputNames);

% 3. 调出图形化设计器进行层拓扑结构与权重可视化检查
deepNetworkDesigner(net);
```

> [!TIP]
> 支持 `importNetworkFromPyTorch` 的版本还可导入符合该版本要求的已导出模型，但 `.pt` 后缀并不表示任意 `state_dict` 都能直接导入。针对支持 traced 模型的版本，可在 Python 使用 `torch.jit.trace(model, dummy_input).save('model.pt')` 后再导入：
> ```matlab
> net = importNetworkFromPyTorch('model.pt');
> ```
> 此路线另需 PyTorch 转换支持包。R2026a 起还有特定 PyTorch 版本的 `ExportedProgram` 支持；请按安装版本的官方兼容表选择格式，不能只复制文件后缀。

---

## 第三步：数据对齐与推理调用——避开张量维度大坑

这是跨框架移植中最容易报错的一环。不同框架对四维图像张量的默认存储顺序不同：

- **PyTorch 规范 (NCHW)**：`[批次 Batch, 通道 Channel, 高度 Height, 宽度 Width]`
- **MATLAB 图像规范 (SSCB)**：`[高度 Spatial, 宽度 Spatial, 通道 Channel, 批次 Batch]`

### 正确的数据对齐推理范例：

```matlab
% 1. 读取一张本地测试图像
img = imread('peppers.png');
assert(size(img, 3) == 3, '本例需要 RGB 三通道图像');
% 对应 ResNet18 DEFAULT 权重：短边缩放到 256，再中心裁剪 224
resize_scale = 256 / min(size(img, 1), size(img, 2));
resized_size = round([size(img, 1), size(img, 2)] * resize_scale);
img_resized = imresize(img, resized_size, 'bilinear');
top = floor((size(img_resized, 1) - 224) / 2) + 1;
left = floor((size(img_resized, 2) - 224) / 2) + 1;
img_resized = img_resized(top:top+223, left:left+223, :);

% 2. 归一化并转为 single 浮点类型
img_single = single(img_resized) / 255.0;
channel_mean = reshape(single([0.485, 0.456, 0.406]), 1, 1, 3);
channel_std = reshape(single([0.229, 0.224, 0.225]), 1, 1, 3);
img_single = (img_single - channel_mean) ./ channel_std;

% 3. 将 MATLAB 的 [224, 224, 3] 转换为符合 ONNX 规范的 dlarray
% 使用 'SSCB' 格式标注（Spatial, Spatial, Channel, Batch）
dl_input = dlarray(img_single, 'SSCB');

% 4. 高效批量推断
scores = predict(net, dl_input);

% 5. ResNet 的原始输出是 logits；显式 Softmax 后才是一组归一化得分
logits = extractdata(scores);
probabilities = exp(logits - max(logits(:)));
probabilities = probabilities / sum(probabilities(:));
[max_prob, class_idx] = max(probabilities(:));
fprintf('最高得分类别索引: %d (Softmax 值: %.4f)\n', class_idx, max_prob);
```

这些索引需要与 Python 权重的类别列表对应；Softmax 数值也不等于已校准的置信度。两端图像插值实现可能有小差异；严谨比较时应保存同一份预处理张量，检查逐元素输入及输出误差。

---

## 第四步：在 Simulink 中实现“AI + 物理系统”闭环仿真

将导入的 `net` 放入控制回路中：
1. 打开任意 Simulink 模型，在 Library 中搜索 **Deep Learning Toolbox**；
2. 拖入 **Predict** 模块（或 **Stateful Predict**）；
3. 双击模块，在 `Network object in base workspace` 输入框中填入我们在 MATLAB 工作区建立的变量名 `net`；
4. 将传感器采集的物理信号连接至该模块输入端，将预测的控制量连接至执行机构动力学模块；
5. 点击运行，即可实时检验神经网络作为控制器时的系统稳定裕度。

---

## 第五步：从 MATLAB 导出回 ONNX

如果你在 MATLAB 内部训练好了前沿网络，希望交付给 Python 团队：

```matlab
% 将 MATLAB 工作区中的 dlnetwork 导出为标准 ONNX
exportONNXNetwork(net, 'matlab_trained_model.onnx', 'OpsetVersion', 17);
disp('已成功导出 ONNX 模型包');
```

---

## 常见错误与排查

### 1. 报错 `Unsupported ONNX operator: ...`
- **原因**：导出的 ONNX 模型中包含了超高版本或自定义的专有算子。
- **解决**：在 PyTorch 导出时指定成熟兼容的 `opset_version=15` 或 `17`，或在 MATLAB 官方附加功能中心更新最新的 `Deep Learning Toolbox Converter for ONNX Model Format` 插件。

### 2. 预测结果完全错乱，置信度全集中在第一项
- **原因**：张量通道顺序颠倒（RGB 与 BGR 混淆），或者输入数据未做均值/方差标准化（ImageNet 标准归一化）。
- **解决**：检查 Python 端的预处理流程（`transforms.Normalize`），确保在传入 MATLAB 之前减去相同均值并除以标准差。

---

## 重要边界

- 导入非常复杂的动态控制流网络（如带动态循环展开的特定 Transformer 采样层）时，建议优先通过 ONNX 导出为固定序列长度的静态图。
- 运行本教程中的导入导出功能需要计算机安装了 **Deep Learning Toolbox**。

---

## 权威参考

- [MathWorks 官方：导入 ONNX 模型文档](https://www.mathworks.com/help/deeplearning/ref/importonnxnetwork.html)
- [MathWorks 官方：importNetworkFromONNX](https://www.mathworks.com/help/deeplearning/ref/importnetworkfromonnx.html)
- [Torchvision ResNet18 权重与预处理](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [MathWorks 官方：PyTorch 模型导入指南](https://www.mathworks.com/help/deeplearning/ref/importnetworkfrompytorch.html)
- [ONNX 官方规范与 GitHub 仓库](https://github.com/onnx/onnx)

## 下一步

- [MATLAB 深度学习现代工作流与旧案例迁移](../Neural%20Network/MODERN_WORKFLOW_GUIDE.md)
- [MATLAB 现代计算案例](README.md)
- [Simulink RC 滤波电路仿真](simulink_rc_circuit_guide.md)
- [MATLAB Coder 入门：从 M 代码到 C/C++ 源码](matlab_coder_intro.md)
