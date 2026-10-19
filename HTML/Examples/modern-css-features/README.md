# 现代 CSS 新特性实验台

这个案例展示了无需任何预处理器（Sass/Less）或前端打包工具（Webpack/Vite），仅凭现代浏览器原生标准即可实现的 CSS 新特性：**原生 CSS 嵌套（Native CSS Nesting）** 与 **父级关系选择器 `:has()`**。

## 适合人群

- 习惯了在 Sass/SCSS 中写 `&` 嵌套，想摆脱额外编译工具的前端开发者。
- 曾经为“CSS 无法根据子元素状态反向修改父元素样式”感到困扰的学习者。
- 想学习用纯 CSS 驱动深色主题切换、表单状态联动等高级无 JS 交互的初学者。

## 学习目标

完成本案例后，你应该能够：

- 在纯 CSS 中熟练使用 `&` 嵌套选择器和媒体查询。
- 使用 `:has()` 实现父元素基于子元素状态（如 `:checked`、`:invalid`）的动态样式联动。
- 理解现代 CSS Baseline 标准以及在生产环境中的兼容性考量。

## 文件说明

| 文件 | 职责 |
| :--- | :--- |
| [index.html](index.html) | 单文件演示页面，包含内嵌样式、语义化 HTML5 结构与实时交互组件 |

## 如何打开

直接在文件资源管理器中双击 [index.html](index.html)，或在任意支持现代标准的浏览器（Chrome 120+、Firefox 121+、Safari 17.2+、Edge 120+）中打开即可直接运行，无需启动任何本地开发服务器。

## 观察顺序

1. **观察右上角“切换深色模式”**：点击切换框，观察背景与文字颜色的平滑转变。检查代码中 `body:has(#dark-mode-toggle:checked)`，体会无需任何 JS 监听即可实现整页主题切换。
2. **观察表单实时校验高亮**：在输入框中随便输入非邮箱字符（如 `abc`），点击框外，观察外层 `.form-group` 容器自动变为红色警示并展示提示。
3. **观察卡片勾选展开**：勾选“激活该项目”复选框，观察虚线边框变为绿色实线，底部操作面板自动显现。
4. **阅读样式代码中的嵌套结构**：查看 `.feature-card` 内部如何直接嵌套 `& .card-header` 与 `@media`。

## 推荐修改点

- 尝试为表单增加一个 `input:valid` 的状态匹配，使得用户输入合法邮箱时父容器边框变绿。
- 尝试使用 `:has()` 编写一个纯 CSS 驱动的下拉手风琴（Accordion）折叠面板。

## 权威参考

- [MDN: CSS nesting](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_nesting)
- [MDN: :has() relational pseudo-class](https://developer.mozilla.org/en-US/docs/Web/CSS/:has)
- [Web.dev: CSS nesting specification and examples](https://web.dev/articles/css-nesting)

## 下一步

- [现代 Web API 实验台](../modern-web-apis/README.md)
- [原生 UI 2026 实验台](../native-ui-2026/README.md)
- [现代 JavaScript ES2024-2026 新特性](../es2024-2026-features/README.md)
