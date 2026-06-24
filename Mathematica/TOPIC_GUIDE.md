# Mathematica 专题学习指南

Mathematica / Wolfram Language 很适合按专题学习。不要一开始追求记住大量函数，先按“符号计算、微积分、线性代数、绘图、规则替换”几个主题建立直觉。

## 专题一：表达式和函数调用

核心记忆：

```wolfram
Function[argument]
```

示例：

```wolfram
Sin[x]
Expand[(x + 1)^3]
Plot[Sin[x], {x, 0, 2 Pi}]
```

学习重点：

- 函数调用用 `[]`。
- 列表用 `{}`。
- 改变优先级用 `()`。

## 专题二：规则替换

规则替换是 Mathematica 的高频用法。

```wolfram
expr = x^2 + 2 x + 1;
expr /. x -> 3
```

理解方式：

- `x -> 3` 是一条规则。
- `/.` 表示把规则应用到表达式。

这和很多语言里的“给变量赋值”不是一回事。

## 专题三：符号化简和方程求解

常用函数：

```wolfram
Expand[(x + 1)^3]
Factor[x^2 - 1]
Simplify[Sin[x]^2 + Cos[x]^2]
Solve[x^2 == 4, x]
```

注意：方程使用 `==`，不是 `=`。

## 专题四：微积分

常用函数：

```wolfram
D[Sin[x], x]
Integrate[x^2, x]
Limit[Sin[x]/x, x -> 0]
```

学习建议：

- 先做符号结果。
- 再用 `N` 转成数值结果。
- 最后画图验证直觉。

## 专题五：线性代数

矩阵通常写成嵌套列表：

```wolfram
A = {{1, 2}, {3, 4}};
Det[A]
Inverse[A]
Eigenvalues[A]
```

如果结果很复杂，可以用：

```wolfram
N[Eigenvalues[A]]
```

## 专题六：绘图

常用绘图：

```wolfram
Plot[Sin[x], {x, 0, 2 Pi}]
ListPlot[{1, 3, 2, 5}]
ContourPlot[x^2 + y^2, {x, -2, 2}, {y, -2, 2}]
```

绘图时最常见的问题是忘记变量范围，或者变量已经被赋值。必要时先 `Clear[x]`。

## 专题七：Notebook 写法

一个好的 Notebook 不只是代码堆叠，而是：

```text
问题说明
-> 公式或数据
-> 计算代码
-> 图形或结果
-> 解释和结论
```

建议每段代码前后都写一句说明。以后复盘时，你会感谢现在的自己。

## 和其他工具对照

| 任务 | Mathematica 更适合 | Matlab / Python 更适合 |
| :--- | :--- | :--- |
| 公式推导 | 是 | 需要额外库或工具箱 |
| 快速画数学函数 | 是 | 也可以，但步骤略多 |
| 工程算法批量实验 | 可以 | Matlab / Python 更常见 |
| 通用自动化脚本 | 不优先 | Python 更适合 |

## 下一步

先从 [基础教程](Basics/README.md) 里选择一个专题跑通，再把同一个问题用 Matlab 或 Python 对照实现。对照越多，越能理解每个工具的性格。
