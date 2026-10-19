# MATLAB Coder 入门：从 M 代码到 C/C++ 独立源码

在科学计算和算法研发中，MATLAB 以其强大的矩阵表达和便捷的调试体验成为算法原型的首选语言。然而，当算法需要被移植到单片机（MCU）、数字信号处理器（DSP）、自动驾驶控制器或大型 C++ 工业软件中运行时，手动用 C 语言重写一遍算法不仅极其费时，而且极易引入边界越界、内存泄漏等隐蔽 Bug。

**MATLAB Coder** 能够将符合特定子集规范的 MATLAB 算法代码（`.m`）直接自动化编译生成**高可读、无任何 MATLAB 运行时依赖的纯独立 C/C++ 源码**，或者编译为本地 **MEX 动态加速库**，让耗时的复杂循环提速数倍至数十倍。

---

## 适合人群

- 已经在 MATLAB 中完成算法原型验证，需要将其部署到嵌入式硬件的工程人员。
- 参加数学建模、RoboMaster 或电子设计竞赛，需要将滤波与控制算法烧录进单片机的学生。
- 经常遇到 MATLAB 双重 `for` 循环耗时过长，希望利用 MEX 获得底层 C 语言运行速度的开发者。

## 学习目标

完成本章后，你应该能够：

- 深刻理解解释型语言（动态类型）与编译型 C/C++ 语言（静态类型）在代码生成时的根本差异。
- 在函数顶部正确声明 `%#codegen` 编译指令并使用代码检查分析器。
- 遵循变量预分配与静态类型约束，编写符合代码生成子集的 MATLAB 函数。
- 使用 `codegen` 命令生成本地 MEX 加速函数，并进行运行耗时实测对比。
- 一键生成可在任意标准 C/C++ 编译器（GCC、Clang、MSVC）下脱机编译的独立 `.c` 和 `.h` 源码。

---

> [!IMPORTANT]
> **前置工具箱依赖检测：**
> `codegen` 命令并不是 MATLAB 基础包自带的内置函数，它隶属于 **MATLAB Coder 工具箱**。
> 在运行本教程前，请先在 MATLAB 命令窗口输入以下命令检测是否已安装授权：
> ```matlab
> hasCoder = license('test', 'Matlab_Coder');
> if ~hasCoder, disp('⚠️ 当前环境未安装 MATLAB Coder 工具箱，请在 MATLAB 顶部【主页】->【附加功能】中搜索安装'); end
> ```
> 若未安装该工具箱直接执行 `codegen` 会报 `未定义函数或变量 'codegen'`。此外，生成 MEX 加速库还需要机器已配置支持的 C 编译器（可在命令行运行 `mex -setup` 验证）。

---

## 代码生成的核心约束与思维转换

动态 MATLAB 代码想要转换为严谨的 C 语言，必须消除所有的“动态模糊性”：

| 特性 | 普通 MATLAB 脚本 | MATLAB Coder 生成子集 |
| :--- | :--- | :--- |
| **变量类型** | 可随意改变（如 `a = 1; a = 'hello';`） | **强类型固定**，同一个变量类型不可二次篡改 |
| **内存分配** | 数组可在循环中随意变长 `arr(end+1) = x` | **必须显式预分配**（使用 `zeros`），尺寸需已知或有上限 |
| **界面交互** | 含有 `plot`、`figure`、`disp` 等函数 | **代码生成严格剔除图形显示**（仅保留纯数学逻辑） |
| **编译指令** | 无 | 函数首行务必标注 `%#codegen` |

---

## 实战案例：移动平均平滑滤波算法

我们以一个经典的滑动平均滤波函数为例，展示从编写到编译的全流程。

### 第一步：编写符合规范的算法函数 `moving_average.m`

```matlab
function y = moving_average(x, window_size)
%#codegen
% MOVING_AVERAGE 针对一维时序信号的平滑滤波函数
% 标注 %#codegen 指令，通知 MATLAB 静态检查器按 C 语言标准严格审查代码

% 1. 显式断言输入维度与类型约束
assert(isa(x, 'double'));
assert(isa(window_size, 'double'));
assert(isscalar(window_size));
assert(window_size >= 1);

N = length(x);
% 2. 核心规范：预分配输出内存空间，避免动态增减内存碎片
y = zeros(1, N);

w = floor(window_size);
for i = 1:N
    % 确定滑动窗口边界
    start_idx = max(1, i - w + 1);
    
    % 计算局部累加均值
    accum_sum = 0.0;
    count = 0;
    for j = start_idx:i
        accum_sum = accum_sum + x(j);
        count = count + 1;
    end
    y(i) = accum_sum / count;
end

end
```

---

## 第二步：生成 MEX 动态链接库进行仿真加速

MEX（MATLAB Executable）是编译好的本地二进制库，能在 MATLAB 内部直接被调用，同时具备底层 C 语言的速度。

在命令行输入：

```matlab
% 定义输入参数的尺寸和数据类型模板 (长度为 10000 的 double 向量，以及标量窗口)
sample_input = zeros(1, 10000);
sample_window = 10.0;

% 执行 MEX 编译命令
codegen -config:mex moving_average -args {sample_input, sample_window}
```

编译完成后，当前目录下会生成 `moving_average_mex.mexw64`（Windows）或 `.mexmaci64`（Mac）。

### 性能加速对比实测：

```matlab
% 构造测试数据
raw_signal = randn(1, 100000);
win = 50.0;

% 测试原生 M 代码耗时
tic;
res_m = moving_average(raw_signal, win);
t_m = toc;

% 测试编译后的 MEX 耗时
tic;
res_mex = moving_average_mex(raw_signal, win);
t_mex = toc;

fprintf('原生 M 代码运行耗时: %.4f 秒\n', t_m);
fprintf('MEX 加速后运行耗时:  %.4f 秒\n', t_mex);
fprintf('提速倍率: %.1fx\n', t_m / t_mex);
assert(max(abs(res_m - res_mex)) < 1e-12, '结果精度校验一致！');
```

---

## 第三步：生成脱离 MATLAB 的独立 C 源码

若你的最终目标是将算法部署到嵌入式单片机或 Linux 服务器，应生成独立的 C 源代码包：

```matlab
% 创建独立 C 语言静态库配置对象
cfg = coder.config('lib');
cfg.TargetLang = 'C';
cfg.GenerateReport = true; % 生成 HTML 代码审查报告

% 执行源码生成
codegen -config cfg moving_average -args {zeros(1, 1000), coder.Constant(5.0)}
```

### 生成结果查看

MATLAB 会在 `codegen/lib/moving_average/` 目录下生成：
- `moving_average.c`：标准 ANSI C 实现的算法逻辑；
- `moving_average.h`：导出的函数声明头文件；
- `rtwtypes.h`：标准跨平台数据类型定义（`real_T`、`boolean_T` 等）；
- `html/index.html`：**交互式代码审查报告**，点击 C 代码的某一行可高亮定位到原始对应的 MATLAB 代码行。

---

## 常见错误与排查

### 1. 报错 `Variable '...' is not fully defined on all execution paths`
- **原因**：C 语言要求所有变量在使用前必须被赋值。若在 `if-else` 分支中只在某一个条件分支给变量赋了初值，代码生成器会报错。
- **解决**：在进入条件判断之前，统一为该变量赋初值（如 `val = 0.0;`）。

### 2. 报错 `Function 'plot' not supported for code generation`
- **原因**：代码中混入了仅用于人机交互的可视化函数。
- **解决**：使用 `coder.extrinsic('plot')` 声明该函数仅用于 MATLAB 内部仿真时不被编译成 C，或者在发布生产代码前将绘图语句剥离出算法核心。

---

## 重要边界

- 本教程演示的是基础算法的代码生成。若涉及复杂的动态可变尺寸矩阵，需使用 `coder.typeof(0, [1, Inf], [0, 1])` 进行显式变长绑定。
- 生成代码的性能与 MATLAB 内置的专用矩阵库（BLAS/LAPACK）在超大矩阵乘法上有细微差异，小型/嵌入式循环场景提升最为显著。

---

## 权威参考

- [MathWorks 官方：MATLAB Coder 入门指南](https://www.mathworks.com/help/coder/getting-started-with-matlab-coder.html)
- [MathWorks 官方：支持代码生成的 MATLAB 语言子集](https://www.mathworks.com/help/coder/ug/functions-supported-for-code-generation--alphabetical-list.html)

## 下一步

- [MATLAB 基础脚本与函数](../Basics/README.md)
- [Simulink RC 滤波电路仿真](simulink_rc_circuit_guide.md)
- [MATLAB 与 PyTorch / ONNX 模型互通](matlab_onnx_interop.md)
