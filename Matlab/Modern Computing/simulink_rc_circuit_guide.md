# Simulink 入门：RC 滤波电路仿真

很多初学者学习 MATLAB 时，习惯在编辑器中一行一行编写算法脚本（`.m` 文件）。但当面对控制工程、无人机姿态解算、汽车悬挂动力学或电力电子系统等复杂的**连续时间动态系统**时，直接手写微分方程不仅容易出错，也很难直观观察多变量间的反馈流动。

**Simulink** 是 MathWorks 提供的图形化动态系统建模与仿真环境。它将复杂的常微分方程（ODE）求解器隐藏在底层，允许工程师像搭积木一样通过“连接模块框图（Block Diagram）”来直观设计、调试和分析物理系统。

---

## 适合人群

- 已经掌握 MATLAB 基础变量与矩阵，希望拓展动态系统仿真能力的学习者。
- 自动化、电气工程、机械电子等专业需要完成课程设计或控制仿真的学生。
- 想理解图形化信号流（Signal Flow）与时间步长求解器如何协同工作的初学者。

## 学习目标

完成本教程后，你应该能够：

- 解释 Simulink 的运行本质（时间步进常微分方程 ODE 求解器 + 数据流驱动框图）。
- 熟练使用四大类高频核心基础模块（Sources 信号源、Sinks 示波器、Continuous 连续算子、Math Operations 数学计算）。
- 在 Simulink 中从零搭建一个一阶 RC 滤波充电电路的完整负反馈回路。
- 在示波器（Scope）中观察时间常数 $\tau = RC$ 下的电容电压上升曲线。
- 编写等效的 MATLAB 脚本，验证仿真输出与数学解析解的精确一致性。

---

> [!IMPORTANT]
> **前置组件确认：**
> Simulink 是 MathWorks 独立的产品组件。在安装 MATLAB 时，若未勾选 Simulink，在命令行输入 `simulink` 会报错 `未定义函数或变量 'simulink'`。
> 可在 MATLAB 命令行执行：
> ```matlab
> hasSimulink = ~isempty(ver('simulink')) && license('test', 'Simulink');
> if ~hasSimulink, disp('需要安装 Simulink，并确认对应许可证可用'); end
> ```
> 确认环境就绪后再进行后续建模实战。

---

## 系统物理背景与数学模型

考虑一个由电阻 $R$ 和电容 $C$ 组成的最简一阶低通滤波电路，输入电压为阶跃阶跃信号 $u(t)$，电容两端输出电压为 $v_c(t)$：

$$u(t) - v_c(t) = i(t) R = R C \frac{d v_c(t)}{d t}$$

整理可得状态方程：

$$\frac{d v_c(t)}{d t} = \frac{1}{R C} \left[ u(t) - v_c(t) \right]$$

两边对时间求积分，即可求出任意时刻的电容电压：

$$v_c(t) = \int \frac{1}{R C} \left[ u(t) - v_c(t) \right] dt$$

### 框图数据流拓扑

```text
[Step 阶跃输入 u(t)]
         │
         ▼ (+)
     ┌───────┐      差值误差 e(t)      ┌───────────┐    导数 dv/dt    ┌───────────┐
     │  Sum  │ ─────────────────────> │ Gain 增益 │ ───────────────> │Integrator │ ───┬───> [Scope 示波器]
     └───────┘                        │  (1/RC)   │                  │  积分器   │    │
         ▲ (-)                        └───────────┘                  └───────────┘    │
         │                                                                            │
         └────────────────────────── 负反馈环路 vc(t) ────────────────────────────────┘
```

---

## 核心模块库速查

| 库类别 (Library) | 常用模块名称 | 作用与功能 |
| :--- | :--- | :--- |
| **Sources (信号源)** | `Step` | 生成阶跃测试信号（可设定跳变时间与起始/终止幅值） |
| | `Sine Wave` | 生成正弦交流扰动测试信号 |
| | `Clock` | 读取仿真当前时间戳 $t$ |
| **Sinks (输出汇流)** | `Scope` | 虚拟示波器，实时绘制变量随时间变化的动态波形 |
| | `To Workspace` | 将仿真计算得到的时序数据直接导出到 MATLAB 工作区 |
| **Continuous (连续系统)** | `Integrator` | 核心积分环节 $1/s$，求解微分方程的核心动力 |
| | `Transfer Fcn` | 直接以传递函数分子分母多项式构建系统模型 |
| **Math Operations (数学)** | `Gain` | 标量或矩阵增益乘法器（如本例中的系数 $1/RC$） |
| | `Sum` | 加法与减法求和节点（支持配置 `+ -` 极性） |

---

## 手把手搭建步骤

1. **启动环境**：在 MATLAB 命令行窗口输入 `simulink` 打开向导，点击 **Blank Model** 创建新工程（或在命令行直接执行 `new_system('rc_filter_demo'); open_system('rc_filter_demo');`）。
2. **放置输入与求和模块**：
   - 从 Library Browser 的 `Simulink/Sources` 拖拽一个 **Step** 模块，设置 `Step time = 0`、`Initial value = 0`、`Final value = 1`；默认阶跃时刻为 1 秒，晚于本例的结束时间，会导致全程零输出；
   - 从 `Simulink/Math Operations` 拖拽一个 **Sum** 模块，双击将参数 `List of signs` 修改为 `|+-`（代表一加一减）。
3. **放置增益与积分模块**：
   - 拖拽 **Gain** 模块，双击将其参数 `Gain` 设置为 `1/(1000 * 1e-6)`（代表 $R=1\text{k}\Omega, C=1\mu\text{F}$）；
   - 从 `Simulink/Continuous` 拖拽 **Integrator** 模块（初始条件保持默认 `0`）。
4. **放置示波器并完成连线**：
   - 拖拽 **Scope** 模块；
   - 连线：`Step` 输出连至 `Sum` 的 `+` 端；`Sum` 输出连至 `Gain` 输入；`Gain` 输出连至 `Integrator` 输入；
   - `Integrator` 输出连至 `Scope`；同时从积分器输出引出一根分支线回连至 `Sum` 的 `-` 端，形成完整的闭环负反馈。
5. **设置求解器并运行**：
   - 在模型上方工具栏将 **Stop Time** 设为 `0.01` 秒（即 10 毫秒）；
   - 在模型设置中选择变步长 `ode45`，将 `Max step size` 设为 `1e-5` 秒，以解析 1 毫秒时间常数；
   - 点击绿色运行按钮 **Run**；
   - 双击 **Scope** 查看示波器，观察电压从 0V 沿指数规律爬升至 1V 的充电响应曲线。

---

## 等效 MATLAB 脚本对照验证

下面使用相同方程、零初值和从 `t=0` 开始的单位阶跃，比较解析解与欧拉近似。欧拉法与变步长 Simulink 求解器的数值轨迹不会逐点完全相同：

```matlab
% RC_FILTER_SCRIPT 对照验证脚本
R = 1000;         % 电阻: 1000 欧姆
C = 1e-6;         % 电容: 1 微法
tau = R * C;      % 时间常数: 0.001 秒 (1ms)

% 时间轴: 0 到 5 倍时间常数 (充电基本完成达到 99.3%)
t = linspace(0, 5 * tau, 500);

% 1. 数学解析精确解
v_analytic = 1.0 * (1 - exp(-t / tau));

% 2. 模拟欧拉前向数值步进积分 (等效固定步长求解器)
dt = t(2) - t(1);
v_sim = zeros(size(t));
for k = 1:(length(t)-1)
    % 导数: dv/dt = (u - v) / (R*C)
    u_in = 1.0;
    dv_dt = (u_in - v_sim(k)) / tau;
    v_sim(k+1) = v_sim(k) + dv_dt * dt;
end

% 绘图对照
figure;
plot(t * 1000, v_analytic, 'b-', 'LineWidth', 2); hold on;
plot(t * 1000, v_sim, 'r--', 'LineWidth', 1.5);
grid on;
xlabel('时间 t (毫秒)');
ylabel('电容电压 vc (V)');
title('RC 滤波电路阶跃响应: 解析解 vs 步进仿真');
legend('解析解 (1 - e^{-t/\tau})', '数值积分模拟', 'Location', 'SouthEast');
```

---

## 推荐修改点

- **修改电阻电容值**：将 $R$ 增大 5 倍，观察时间常数 $\tau$ 变大后示波器上升曲线如何变得平缓。
- **替换信号源**：将 `Step` 模块删除，替换为 `Sine Wave`（正弦波信号源），观察高频输入信号如何被低通滤波器滤除、幅值大幅衰减。
- **升级二阶系统**：串联两级 RC 积分回路，观察二阶系统的相位滞后与延迟响应。

---

## 重要边界

- 本教程演示的是基于数学框图（Signal Flow）的抽象仿真，信号连线代表单纯的“数值传递”。
- 若需要直接画电阻、电容物理器件引脚和接地元器件，应使用专业的物理网络工具箱 **Simscape / Simscape Electrical**。
- 本案例未涉及 C 语言代码生成（Embedded Coder）和实时硬件回路（HIL）。

---

## 权威参考

- [MathWorks 官方：Simulink 入门指南](https://www.mathworks.com/help/simulink/getting-started-with-simulink.html)
- [MathWorks 官方：选择常微分方程 (ODE) 求解器](https://www.mathworks.com/help/simulink/gui/solver.html)

## 下一步

- [MATLAB 基础教程](../Basics/README.md)
- [MATLAB 现代计算案例](README.md)
- [MATLAB Coder 入门：从 M 代码到 C/C++ 源码](matlab_coder_intro.md)
