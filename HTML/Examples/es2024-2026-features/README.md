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
| 1 | Set 集合方法 | ES2025 | `union`, `intersection`, `difference`, `symmetricDifference` |
| 2 | Promise.withResolvers | ES2024 | `Promise.withResolvers()` |
| 3 | Iterator Helpers | ES2025 | `.map()`, `.filter()`, `.take()`, `.toArray()` |
| 4 | Object.groupBy | ES2024 | `Object.groupBy()` |

## 浏览器兼容性

语言规范年份与浏览器首次支持年份不同；`Set` 新方法进入 ES2025，不能因为浏览器在 2024 年实现就标成 ES2024。兼容性以页面特性检测和下方各官方参考页的兼容表为准，不凭版本号推断所有平台都可用。

页面检测实际使用的 Iterator 方法。不支持时显示提示。斐波那契使用 `BigInt` 保持大整数精确，输入限制为 1 到 50，避免错误输入导致超长计算。

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

1. 在开发者工具中给 `fibonacci()` 的 `yield` 加断点，观察 `.take()` 按需请求下一项。
2. 在控制台运行 `Map.groupBy(products, p => p.category)`，与现有 `Object.groupBy()` 结果对照。
3. 修改 `products` 中的库存和类别，再点击分组按钮观察变化。
4. 将每个 Set 集合的输入改为重复字符串，观察集合去重和插入顺序。

原扩展方向仍可从现有资料定位：源码展开可参考 [交互组件](../interactive-components/README.md) 的折叠面板；`Map.groupBy()` 见本例控制台对照；日期时间 API 可从 [TC39 提案列表](https://github.com/tc39/proposals) 查询当前状态。性能测量应单独设计基准，当前页面不宣称已做内存性能比较。

## 下一步

- [HTML 基础示例目录](../README.md)
- [TC39 Stage 3 提案](https://github.com/tc39/proposals#stage-3) — 即将到来的特性预览
