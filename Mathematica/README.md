# Mathematica 学习笔记

这个目录保存 Mathematica / Wolfram Language 基础教程和命令速查笔记。

## 推荐入口

- [Mathematica 学习导读](LEARNING_GUIDE.md)
- [Mathematica 常见问题](FAQ.md)
- [Mathematica 专题学习指南](TOPIC_GUIDE.md)
- [Mathematica Notebook 写作指南](NOTEBOOK_GUIDE.md)
- [Mathematica 基础教程](Basics/README.md)
- [Mathematica 可运行案例](Examples/README.md)
- [Mathematica.md](Mathematica.md)：旧版命令速查笔记

## 基础教程内容

[Basics](Basics/README.md) 面向新手，包含：

- Mathematica 是什么与基本操作
- 表达式、变量与函数调用
- 列表、规则与常用操作
- 符号计算与方程求解
- 矩阵与线性代数
- 微积分与数值计算
- 绘图与可视化
- 文件导入导出与练习路线

## 可运行案例

[Examples](Examples/README.md) 把教程概念组合成完整 Wolfram Language 脚本：

- 图社区发现与可视化
- 符号化物理约束损失
- 高层神经网络分类流程
- [神经网络梯度推导与中心差分校验](Examples/GRADIENT_CHECK_GUIDE.md)

建议先在 Notebook 中逐段运行，再尝试使用配置好 Kernel 的 `wolframscript` 执行整个 `.wl` 文件。

## 学习建议

Mathematica 很适合做符号推导、数学实验和快速可视化。新手建议先熟悉：

- 函数调用格式：`Function[argument]`
- 列表格式：`{1, 2, 3}`
- 替换规则：`x -> value`
- 精确计算和数值计算的区别
- 赋值 `=` 和方程 `==` 的区别

如果你只是想快速查某个命令，可以看 [Mathematica.md](Mathematica.md)。

如果你想先建立学习路线、理解规则替换、符号计算、绘图和与 Matlab/Python 的区别，可以先看 [Mathematica 学习导读](LEARNING_GUIDE.md)。掌握基础后，再用 [可运行案例](Examples/README.md) 把零散命令连成完整流程。
