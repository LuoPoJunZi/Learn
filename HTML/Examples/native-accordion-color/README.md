# 原生互斥手风琴与 oklch() / color-mix() 色彩实验台

## 适合人群与学习目标

适合已了解基础 HTML 表单与 CSS 变量，想减少冗余 JavaScript 折叠逻辑并学习现代感知均匀色彩系统的初学者。

资料核验：**2026-11-20**，参考 MDN `<details>`、`oklch()` 与 `color-mix()` 文档。本案例包含 sRGB 基础回退与 `@supports` 渐进增强，在旧浏览器中依然可读可用。

完成后你应该能够：

- 使用原生 `<details name="...">` 实现零 JS 互斥折叠手风琴。
- 解释 `oklch()` 相比 `hsl()` 在感知亮度一致性与无障碍对比度上的优势。
- 使用 `color-mix()` 基于单一主色动态派生浅背景色与深强调色。
- 用 `@supports` 为现代 CSS 色彩函数保留可靠的 sRGB 回退。

## 文件说明

| 文件 | 职责 |
| :--- | :--- |
| [index.html](index.html) | 色相滑块预览区与 3 组 `<details name="faq-group">` 互斥手风琴 |
| [style.css](style.css) | sRGB 基础回退色、`@supports` 渐进增强与焦点样式 |
| [script.js](script.js) | 运行时 `CSS.supports` 检测与色相滑块更新 `--brand-hue` |

## 如何打开与观察

双击 [index.html](index.html) 即可在浏览器中查看，无需构建工具或本地服务器：

1. 拖动色相滑块，观察主色、浅背景色和深强调色同步变化且明暗层次保持稳定。
2. 依次点击下方 3 个手风琴标题，观察同一时间仅展开 1 项（完全没有编写监听点击的 JS 代码）。
3. 使用键盘 `Tab` 与 `Enter` / `Space` 操作手风琴，验证原生键盘无障碍体验。

## 权威参考与下一步

- [MDN：`<details>` 元素与 `name` 属性](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/details)
- [MDN：`oklch()` 色彩函数](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Values/color_value/oklch)
- [MDN：`color-mix()` 函数](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Values/color_value/color-mix)
- [CSS `@scope` 组件样式边界案例](../scoped-css/README.md)
- [JavaScript Temporal API 实验台](../temporal-api/README.md)
