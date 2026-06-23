# Mathematica 学习导读

Mathematica / Wolfram Language 的强项是符号计算、公式推导、可视化和快速数学实验。它和 Python、Matlab 的思维方式不同：很多操作都围绕表达式、规则、模式和函数式计算展开。

## 适合人群

- 想学习符号计算和数学可视化的新手。
- 需要做公式推导、微积分、线性代数的人。
- 已经会 Matlab 或 Python，想了解 Mathematica 思维方式的人。

## 学习目标

学完后，你应该能：

- 看懂 `Function[argument]` 形式的函数调用。
- 理解列表、规则、替换、精确计算和数值计算。
- 做基础符号求解、微积分、矩阵运算。
- 用 `Plot`、`ListPlot`、`Graphics` 做基础可视化。

## 推荐阅读顺序

1. [Mathematica 基础教程](Basics/README.md)
2. [旧版命令速查笔记](Mathematica.md)
3. 本文的专题路线和对照学习说明

## 学习地图

```text
表达式和函数调用
-> 列表和规则
-> 替换与模式
-> 符号计算
-> 数值计算
-> 绘图与可视化
-> 数据导入导出
```

## 核心概念

### 表达式

Wolfram Language 中很多内容都可以看作表达式。函数调用使用方括号：

```wolfram
Sin[x]
Plot[Sin[x], {x, 0, 2 Pi}]
```

### 规则替换

规则写作 `x -> value`，常用于把符号表达式替换成具体值：

```wolfram
x^2 + 2 x + 1 /. x -> 3
```

### 精确计算与数值计算

`1/3` 会保持精确分数，`1.0/3` 会进入近似数值计算。需要数值结果时可以使用 `N`：

```wolfram
N[1/3]
```

## 典型专题

### 符号计算

常用函数：

```wolfram
Expand[(x + 1)^3]
Factor[x^2 - 1]
Solve[x^2 == 4, x]
Simplify[Sin[x]^2 + Cos[x]^2]
```

### 微积分

常用函数：

```wolfram
D[Sin[x], x]
Integrate[x^2, x]
Limit[Sin[x]/x, x -> 0]
```

### 线性代数

常用函数：

```wolfram
A = {{1, 2}, {3, 4}};
Det[A]
Inverse[A]
Eigenvalues[A]
```

### 绘图

常用入口：

```wolfram
Plot[Sin[x], {x, 0, 2 Pi}]
ListPlot[{1, 3, 2, 5}]
ContourPlot[x^2 + y^2, {x, -2, 2}, {y, -2, 2}]
```

## Mathematica 与 Matlab/Python 对照

| 任务 | Mathematica | Matlab | Python |
| :--- | :--- | :--- | :--- |
| 符号计算 | 内置强项 | Symbolic Math Toolbox | SymPy |
| 数值矩阵 | 列表和矩阵函数 | 内置矩阵 | NumPy |
| 绘图 | `Plot`、`Graphics` | `plot` | matplotlib |
| 规则替换 | `/.` 和 `->` | 不常用 | 字典/表达式替换 |
| 交互式笔记 | Notebook | Live Script | Jupyter Notebook |

## 少量自查

- 你能解释 `=` 和 `==` 的区别吗？
- 你知道 `[]`、`{}`、`()` 各自常见用途吗？
- 你能用规则替换给表达式代入数值吗？
- 你能区分精确结果和近似结果吗？

## 外部资源

- [Wolfram Language Documentation](https://reference.wolfram.com/language/)：官方语言文档。
- [An Elementary Introduction to the Wolfram Language](https://www.wolfram.com/language/elementary-introduction/)：Stephen Wolfram 的入门书。
- [Wolfram Fast Introduction](https://www.wolfram.com/language/fast-introduction-for-programmers/)：适合已有编程基础的人。
- [Wolfram Function Repository](https://resources.wolframcloud.com/FunctionRepository/)：可查找社区函数和示例。

## 下一步

建议先把基础教程里的符号计算、矩阵、绘图章节跑一遍，再把同一个数学问题分别用 Mathematica、Matlab、Python 写一版。对照学习会很快暴露三种工具的思维差异。
