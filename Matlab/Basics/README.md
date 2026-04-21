# Matlab 基础

这个目录保存 Matlab 基础学习笔记。

## 文件说明

- [Matlab_study.md](Matlab_study.md)：Matlab 基础语法、变量、矩阵、数组运算等入门内容
- [Matlab_parallel computing.md](Matlab_parallel%20computing.md)：Matlab 并行计算相关内容

## 建议学习顺序

1. 先看 `Matlab_study.md` 中的基础命令。
2. 学习向量、矩阵和数组运算。
3. 学习数据读写和绘图。
4. 有性能需求时，再看并行计算。

## 基础知识地图

### 命令窗口与工作区

常用命令：

```matlab
clc
clear
who
whos
format long
```

### 向量和矩阵

Matlab 的核心是矩阵和数组。重点理解：

- 行向量和列向量
- 矩阵索引
- 冒号运算符
- 矩阵拼接
- 矩阵乘法和点乘
- 转置

### 常用特殊矩阵

```matlab
eye(n)
zeros(m, n)
ones(m, n)
rand(m, n)
linspace(a, b, n)
diag(v)
```

### 并行计算

并行计算适合循环任务、批量实验和计算量较大的算法。使用前需要确认是否安装 Parallel Computing Toolbox。

## 后续优化建议

后续可以把 `Matlab_study.md` 拆成多篇：

- 基础语法
- 向量与矩阵
- 绘图
- 数据读写
- 函数与脚本
- 并行计算
