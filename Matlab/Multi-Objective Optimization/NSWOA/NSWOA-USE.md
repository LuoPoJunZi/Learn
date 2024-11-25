# 多目标鲸鱼优化算法（NSWOA）应用
在将 **多目标鲸鱼优化算法（NSWOA）** 应用于其他实例时，主要需要修改以下几个部分或步骤。不同实例的变化可能涉及目标函数、变量范围、种群规模等关键因素。以下结合代码进行分析：

---

### **1. 修改目标函数**
**文件：`evaluate_objective.m`**

#### 修改内容：
- 更改优化问题的目标函数。例如，将 `zdt1` 替换为其他目标函数（如 `zdt2`、`zdt3` 或自定义函数）。
- 如果目标函数的数量（M）或特性发生变化，需要相应调整代码。

#### 示例：
假设需要优化一个自定义的目标函数，可以修改 `evaluate_objective.m` 为：

```matlab
function f = evaluate_objective(x)
% 自定义目标函数
f(1) = x(1)^2 + sum(x(2:end));      % 目标1
f(2) = (1 - x(1))^2 + sum(x(2:end)); % 目标2
end
```

#### 注意：
- 目标函数的数量（M）需要与 `MainNSWOA.m` 和其他相关文件中指定的值一致。
- 如果是高维目标问题，还需确保绘图函数（如 `plot_data2.m`）适配高维目标的可视化需求。

---

### **2. 修改决策变量的范围**
**文件：`MainNSWOA.m`**

#### 修改内容：
调整决策变量的上下界（`LB` 和 `UB`）来匹配实际问题。例如，变量范围可能不再是 `[0,1]`，而是 `[10, 100]` 或其他。

#### 示例：
如果决策变量范围是 `[10, 100]`，可以修改：

```matlab
LB = ones(1, D) * 10; % 下界
UB = ones(1, D) * 100; % 上界
```

#### 注意：
- `initialize_variables.m` 中的种群初始化代码会自动根据 `LB` 和 `UB` 调整变量范围。

---

### **3. 修改种群规模与迭代次数**
**文件：`MainNSWOA.m`**

#### 修改内容：
根据优化问题的复杂度，调整种群大小（`SearchAgents_no`）和最大迭代次数（`Max_iteration`）。

#### 示例：
对于较复杂的问题，可以增大种群规模和迭代次数：

```matlab
Max_iteration = 500;       % 最大迭代次数
SearchAgents_no = 200;     % 种群规模
```

#### 注意：
- 种群规模与迭代次数会显著影响运行时间和解的质量，需要根据实际问题权衡。

---

### **4. 修改非支配排序逻辑**
**文件：`non_domination_sort_mod.m`**

#### 修改内容：
如果目标函数的数量（M）发生变化，此文件的排序逻辑无需更改，但计算拥挤距离的部分可能需要适配高维目标问题。

#### 示例：
如果目标函数超过 3 个，可以扩展拥挤距离的可视化分析，或对高维数据进行投影。

---

### **5. 修改绘图部分**
**文件：`plot_data2.m`**

#### 修改内容：
当目标函数数量（M）改变时，绘图方式需要调整。默认的 `plot_data2.m` 只能绘制二维目标。如果目标函数数量增加到 3 维，可以修改为：

```matlab
function plot_data2(M, D, Pareto)
pl_data = Pareto(:, D+1:D+M); % 提取目标值
X = pl_data(:, 1);
Y = pl_data(:, 2);
Z = pl_data(:, 3); % 第三个目标值
figure;
scatter3(X, Y, Z, '*', 'k'); % 三维散点图
title('Optimal Solution Pareto Set');
xlabel('Objective function value 1');
ylabel('Objective function value 2');
zlabel('Objective function value 3');
grid on;
end
```

#### 注意：
- 对于高维目标问题（M > 3），无法直接绘图。可通过降维方法（如 PCA）或性能指标分析来展示结果。

---

### **6. 修改约束条件**
**文件：`NSWOA.m`**

#### 修改内容：
如果问题存在约束条件（如等式约束或不等式约束），需要在 `NSWOA.m` 中的解更新步骤添加约束检查。

#### 示例：
假设存在约束条件 `x(1)^2 + x(2)^2 <= 1`，可以在更新解时添加约束检查：

```matlab
% 检查约束
if sum(Whale_posNew1(:,1:2).^2, 2) <= 1
    % 如果满足约束，接受解
    Whale_pos(i,1:K) = Whale_posNew1(:,1:K);
else
    % 如果不满足约束，拒绝解
    Whale_pos(i,1:K) = Whale_pos(i,1:K);
end
```

#### 注意：
- 确保所有约束条件在解生成后进行验证。

---

### **7. 添加新性能指标**
**文件：`MainNSWOA.m`**

#### 修改内容：
如果需要评估算法性能，可以在主文件中添加指标计算代码。例如，计算超体积（HV）或反向生成距离（IGD）。

#### 示例：
```matlab
% 计算 IGD 指标
true_pareto = load('true_pareto.txt'); % 理想帕累托前沿
igd_value = calculate_igd(Pareto(:,D+1:D+M), true_pareto);
fprintf('IGD: %.4f\n', igd_value);
```

#### 注意：
- 指标计算需要理想帕累托前沿数据（如 `true_pareto.txt`）。

---

### **总结：主要修改部分**
1. **目标函数（`evaluate_objective.m`）**：定义新的目标函数。
2. **决策变量范围（`MainNSWOA.m`）**：调整 `LB` 和 `UB`。
3. **种群规模与迭代次数（`MainNSWOA.m`）**：优化运行时间与解的质量。
4. **绘图函数（`plot_data2.m`）**：适配目标函数数量的变化。
5. **约束条件（`NSWOA.m`）**：对解进行约束检查。
6. **性能指标分析（新增模块）**：根据需求计算优化指标。
