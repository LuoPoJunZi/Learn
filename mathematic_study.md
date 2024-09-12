# mathematica
## CMD快速入门
### 1. **基本算术运算**

```mathematica
3 + 5  (* 计算 3 加 5，结果为 8 *)

7 - 2  (* 计算 7 减 2，结果为 5 *)

4 * 6  (* 计算 4 乘以 6，结果为 24 *)

10 / 2  (* 计算 10 除以 2，结果为 5 *)

2^3  (* 计算 2 的 3 次幂，即 2*2*2，结果为 8 *)
```

### 2. **符号计算**

```mathematica
Simplify[Sin[x]^2 + Cos[x]^2]  
(* 简化三角函数表达式，结果为 1，因为 Sin[x]^2 + Cos[x]^2 是三角恒等式 *)

Expand[(x + 1)^2]  
(* 展开二项式 (x + 1)^2，结果为 x^2 + 2 x + 1 *)

Factor[x^2 - 2 x - 3]  
(* 对表达式进行因式分解，结果为 (x - 3)(x + 1) *)

Solve[x^2 - 4 == 0, x]  
(* 求解方程 x^2 - 4 = 0，结果为 x = -2 或 x = 2 *)

DSolve[y'[x] == y[x], y[x], x]  
(* 求解微分方程 y'(x) = y(x)，结果为 y(x) = C1*Exp[x]，C1 为常数 *)
```

### 3. **矩阵运算**

```mathematica
Det[{{1, 2}, {3, 4}}]  
(* 计算 2x2 矩阵的行列式，矩阵为 {{1, 2}, {3, 4}}，结果为 -2 *)

Inverse[{{1, 2}, {3, 4}}]  
(* 计算矩阵的逆矩阵，结果为 {{-2, 1}, {3/2, -1/2}} *)

Transpose[{{1, 2}, {3, 4}}]  
(* 计算矩阵的转置，将行和列互换，结果为 {{1, 3}, {2, 4}} *)

Eigenvalues[{{1, 2}, {3, 4}}]  
(* 计算矩阵的特征值，结果为 {-0.372, 5.372}，特征值表示矩阵的固有值 *)

Eigenvectors[{{1, 2}, {3, 4}}]  
(* 计算矩阵的特征向量，结果为 {{-0.824, 0.416}, {0.416, 0.824}} *)
```

### 4. **微积分运算**

```mathematica
D[x^2 + 3 x + 1, x]  
(* 对 x^2 + 3x + 1 这个多项式对 x 求导，结果为 2x + 3 *)

Integrate[x^2, x]  
(* 计算 x^2 关于 x 的不定积分，结果为 (1/3) x^3，积分常数 C 默认为 0 *)

Integrate[x^2, {x, 0, 1}]  
(* 计算 x^2 在区间 [0, 1] 上的定积分，结果为 (1/3) *)

Limit[Sin[x]/x, x -> 0]  
(* 计算表达式 Sin[x]/x 当 x 趋近于 0 时的极限，结果为 1 *)
```

### 5. **数值计算**

```mathematica
N[Pi]  
(* 将 Pi 转换为近似数值，默认保留 6 位有效数字，结果为 3.14159 *)

FindRoot[Cos[x] - x == 0, {x, 0}]  
(* 使用数值方法求解方程 Cos[x] - x = 0，初值设为 x = 0，结果为 x ≈ 0.739 *)

NSolve[x^3 - x == 0, x]  
(* 数值求解方程 x^3 - x = 0，结果为 {x -> -1., x -> 0., x -> 1.} *)

NIntegrate[Exp[-x^2], {x, 0, 1}]  
(* 数值计算 Exp[-x^2] 在区间 [0, 1] 上的定积分，结果为 0.746824 *)
```

### 6. **绘图与可视化**

```mathematica
Plot[Sin[x], {x, 0, 2 Pi}]  
(* 绘制 Sin[x] 在区间 [0, 2π] 上的图像 *)

Plot3D[Sin[x] Sin[y], {x, 0, Pi}, {y, 0, Pi}]  
(* 绘制三维图像，函数是 Sin[x] Sin[y]，x 和 y 分别在 [0, π] 的区间上变化 *)

ListPlot[{{1, 2}, {2, 3}, {3, 5}, {4, 7}}]  
(* 绘制一组数据点的散点图，数据为 (1, 2), (2, 3), (3, 5), (4, 7) *)

ContourPlot[Sin[x] Sin[y], {x, 0, 2 Pi}, {y, 0, 2 Pi}]  
(* 绘制 Sin[x] Sin[y] 在 [0, 2π] 区域上的等高线图 *)

ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]  
(* 绘制参数曲线，参数为 t，表示单位圆上的坐标 (Cos[t], Sin[t]) *)
```

### 7. **列表与数组**

```mathematica
Table[i^2, {i, 1, 5}]  
(* 生成列表，i 从 1 到 5，每个元素是 i 的平方，结果为 {1, 4, 9, 16, 25} *)

Range[1, 5]  
(* 生成从 1 到 5 的整数序列，结果为 {1, 2, 3, 4, 5} *)

Map[Sin, {0, Pi/2, Pi}]  
(* 将 Sin 函数应用到列表中的每个元素，结果为 {0, 1, 0} *)

Apply[Plus, {1, 2, 3}]  
(* 对列表 {1, 2, 3} 应用加法运算，相当于 1 + 2 + 3，结果为 6 *)
```

### 8. **逻辑与条件运算**

```mathematica
If[x > 0, "Positive", "Non-positive"]  
(* 条件判断：如果 x 大于 0，返回 "Positive"，否则返回 "Non-positive" *)

Which[x < 0, "Negative", x == 0, "Zero", x > 0, "Positive"]  
(* 多重条件判断：根据 x 的值判断是负数、零还是正数，返回相应字符串 *)

And[True, False]  
(* 逻辑与运算：True 和 False 的结果为 False *)

Or[True, False]  
(* 逻辑或运算：True 或 False 的结果为 True *)

Equal[2 + 2, 4]  
(* 判断 2 + 2 是否等于 4，结果为 True *)
```

### 9. **程序结构与控制**

```mathematica
Module[{x = 3}, x^2]  
(* 使用局部变量 x，其值为 3，返回 x^2，结果为 9，局部变量 x 只在 Module 内部有效 *)

Do[Print[i], {i, 1, 5}]  
(* 循环打印 i，从 1 到 5，输出为：1 2 3 4 5 *)

For[i = 1, i <= 5, i++, Print[i]]  
(* 使用 For 循环打印 i，从 1 到 5，输出为：1 2 3 4 5 *)

While[x < 10, x *= 2; Print[x]]  
(* 使用 While 循环，每次将 x 翻倍并打印，直到 x 不小于 10，输出为：2 4 8 16 *)
```

### 10. **文件与输入输出**

```mathematica
Export["data.csv", {{1, 2}, {3, 4}}]  
(* 将二维数组导出为 CSV 文件 "data.csv" *)

Import["data.csv"]  
(* 导入 CSV 文件中的数据，结果为二维数组 {{1, 2}, {3, 4}} *)

Print["Hello, World!"]  
(* 输出字符串 "Hello, World!" 到控制台 *)

InputString["Enter your name: "]  
(* 提示用户输入一个字符串，并返回该字符串 *)

OpenWrite["output.txt"]  
(* 打开一个名为 "output.txt" 的文件用于写入 *)
```

### 11. **随机数与统计**

```mathematica
RandomReal[{0, 1}]  
(* 生成一个在区间 [0, 1] 之间的随机实数，结果为随机值，如 0.678 *)

RandomInteger[{1, 6}]  
(* 生成一个在 1 和 6 之间的随机整数，模拟掷骰子的效果，结果为随机值，如 3 *)

Mean[{1, 2, 3, 4, 5}]  
(* 计算列表中数值的均值，结果为 3 *)

Variance[{1, 2, 3, 4, 5}]  
(* 计算列表中数值的方差，结果为 2 *)

StandardDeviation[{1, 2, 3, 4, 5}]  
(* 计算列表中数值的标准差，结果为 sqrt(2) ≈ 1.414 *)
```

### 12. **字符串处理**

```mathematica
StringJoin["Hello", " ", "World"]  
(* 拼接字符串 "Hello" 和 "World"，中间用空格连接，结果为 "Hello World" *)

StringLength["Mathematica"]  
(* 返回字符串 "Mathematica" 的长度，结果为 11 *)

StringTake["Hello, World!", 5]  
(* 从字符串 "Hello, World!" 中取前 5 个字符，结果为 "Hello" *)

StringReplace["Hello, World!", "World" -> "Mathematica"]  
(* 将字符串 "Hello, World!" 中的 "World" 替换为 "Mathematica"，结果为 "Hello, Mathematica!" *)
```

### 13. **函数定义与操作**

```mathematica
f[x_] := x^2  
(* 定义一个名为 f 的函数，f(x) = x^2 *)

f[3]  
(* 计算函数 f 在 x = 3 处的值，结果为 9 *)

Map[f, {1, 2, 3}]  
(* 将函数 f 应用于列表中的每个元素，结果为 {1, 4, 9} *)

Function[x, x^2][5]  
(* 定义匿名函数，并在 x = 5 时求值，结果为 25 *)
```
