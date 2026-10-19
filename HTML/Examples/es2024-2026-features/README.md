# ES2024-2026 新特性交互演示

> 通过浏览器直接体验最新 JavaScript 语言特性，所有示例均可交互运行。

## 适合人群

- 已掌握 JavaScript 基础，想了解语言最新演进的开发者
- 正在考虑在项目中使用新特性、需要评估兼容性的前端工程师
- 对 TC39 提案流程感兴趣的技术爱好者

## 学习目标

完成本页后，你应该能够：

1. 使用 `Set` 的 `.union()` / `.intersection()` / `.difference()` / `.symmetricDifference()` 进行集合运算
2. 用 `Promise.withResolvers()` 替代传统 Deferred 模式
3. 理解 Iterator Helpers 的惰性求值与数组方法的区别
4. 使用 `Object.groupBy()` 替代手写 `reduce` 分组

## 演示内容

| # | 特性 | 规范版本 | 核心方法 |
|---|------|----------|----------|
| 1 | Set 集合方法 | ES2024 | `union`, `intersection`, `difference`, `symmetricDifference` |
| 2 | Promise.withResolvers | ES2024 | `Promise.withResolvers()` |
| 3 | Iterator Helpers | ES2025 | `.map()`, `.filter()`, `.take()`, `.toArray()` |
| 4 | Object.groupBy | ES2024 | `Object.groupBy()` |

## 浏览器兼容性

| 浏览器 | Set Methods | Promise.withResolvers | Iterator Helpers | Object.groupBy |
|--------|:-----------:|:---------------------:|:----------------:|:--------------:|
| Chrome | 122+ | 119+ | 122+ | 117+ |
| Firefox | 127+ | 121+ | 131+ | 119+ |
| Safari | 17.4+ | 17.2+ | 18.2+ | 17.0+ |
| Edge | 122+ | 119+ | 122+ | 117+ |

> **提示**：页面内置了特性检测，如果浏览器不支持某特性会显示黄色兼容性警告。

## 如何运行

直接在现代浏览器中打开 `index.html` 即可，无需构建工具或服务器：

```bash
# 方式 1：直接双击打开
# 方式 2：命令行
start index.html          # Windows
open index.html           # macOS
xdg-open index.html       # Linux
```

## 权威参考

- [MDN - Set Methods](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set)
- [MDN - Promise.withResolvers()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/withResolvers)
- [MDN - Iterator Helpers](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Iterator)
- [MDN - Object.groupBy()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/groupBy)
- [TC39 提案列表 (Finished Proposals)](https://github.com/tc39/proposals/blob/main/finished-proposals.md)

## 推荐修改点

1. 为每个特性添加「展开源码」折叠区域，展示完整代码片段
2. 增加 `Map.groupBy()` 演示，对比 `Object.groupBy()` 的差异
3. 加入 `Temporal` API（Stage 3）的预览示例
4. 添加性能对比：Iterator Helpers 惰性 vs `Array` 方法的内存差异

## 下一步

- [HTML 基础示例目录](../README.md)
- [TC39 Stage 3 提案](https://github.com/tc39/proposals#stage-3) — 即将到来的特性预览
