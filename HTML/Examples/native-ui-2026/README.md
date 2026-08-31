# 原生 UI 2026 实验台

这个案例展示 2025-2026 年逐步进入现代浏览器的新原生交互能力：Invoker Commands、CSS Anchor Positioning 和 Scroll-driven Animations。页面不依赖框架、npm 或远程 CDN，并为旧浏览器保留基础回退。

## 适合人群

- 已掌握 HTML、CSS 和基础 DOM 操作的学习者
- 写过弹窗、浮层或滚动动画，想减少手工 JavaScript 的人
- 想理解“渐进增强”如何落到真实代码的人
- 准备阅读现代浏览器规范和兼容表的人

## 学习目标

- 使用 `command` 和 `commandfor` 声明按钮与 `<dialog>` 的关系
- 使用 Popover API 与 CSS Anchor Positioning 创建跟随按钮的浮层
- 使用 `scroll()` 和 `view()` 时间线驱动 CSS 动画
- 使用特性检测提供旧浏览器回退
- 使用 `prefers-reduced-motion` 尊重减少动态效果的系统设置

## 文件说明

| 文件 | 作用 |
| :--- | :--- |
| [index.html](index.html) | 对话框、Popover、页面区块与语义结构 |
| [style.css](style.css) | 锚点定位、滚动时间线、响应式布局和减少动态效果 |
| [script.js](script.js) | 特性检测、回退行为和对话框结果反馈 |

## 如何打开

直接双击 [index.html](index.html) 即可。为了看到全部原生效果，建议使用已更新到当前稳定版本的 Chrome、Edge、Firefox 或 Safari。

页面可以直接从本地文件打开。某项 API 不受当前浏览器支持时，“当前浏览器支持状态”会显示“使用回退”。

## 三项重点能力

### Invoker Commands

传统写法需要 JavaScript 查询按钮和对话框，再注册点击事件：

```javascript
button.addEventListener("click", () => dialog.showModal());
```

Invoker Commands 允许按钮直接声明目标和动作：

```html
<button commandfor="release-dialog" command="show-modal">
  打开对话框
</button>
```

当前内置命令可以控制 Dialog 和 Popover，也支持以 `--` 开头的自定义命令。原生语义不会替代业务校验，但能减少常见交互胶水代码。

### CSS Anchor Positioning

过去的浮层定位经常依赖 `getBoundingClientRect()`、滚动监听和窗口变化计算。锚点定位让 CSS 表达“这个浮层跟随哪个元素、优先放在哪里、溢出时如何翻转”。

本案例中 `popovertarget` 建立了按钮与 Popover 的隐式锚点关系，CSS 再指定：

```css
.anchor-note {
  position-area: block-end span-inline-end;
  position-try-fallbacks: flip-block, flip-inline;
}
```

锚点定位的细节仍在快速演进，因此案例把它放进 `@supports`。不支持时，Popover 仍由浏览器居中显示。

### Scroll-driven Animations

页面顶部进度条使用根滚动容器作为时间线：

```css
.reading-progress {
  animation: reading-progress linear;
  animation-timeline: scroll(root block);
}
```

三个原语区块使用 `view()` 时间线，在元素进入视口时逐步完成动画。整个过程不注册 `scroll` 事件。

动画只是增强，不能阻塞阅读和操作。案例在 `prefers-reduced-motion: reduce` 下关闭滚动动画。

## 渐进增强顺序

1. HTML 先提供按钮、Dialog、Popover 和可读内容。
2. CSS 增加锚点位置与滚动动画。
3. JavaScript 检测支持状态。
4. 旧浏览器只补缺失的打开、关闭和浮层显示行为。

回退代码只在缺少原生能力时注册，避免同一次点击同时触发原生动作和 JavaScript 动作。

## 观察顺序

1. 打开声明式对话框，分别选择取消和确认。
2. 查看输出区域是否得到对应的 `returnValue`。
3. 打开兼容提示，调整窗口大小，观察浮层位置。
4. 滚动页面，观察顶部阅读进度和区块入场。
5. 查看支持状态，再到 `script.js` 找对应特性检测。

## 推荐修改点

- 增加一个 `command="request-close"` 按钮，比较它和 `close`。
- 为锚定浮层增加自己的 `@position-try` 方案。
- 把一个 `view()` 动画改成水平方向，并检查窄屏是否溢出。
- 在浏览器开发者工具中模拟 `prefers-reduced-motion`。
- 删除 JavaScript 回退后，在旧版浏览器中比较差异。

## 重要边界

- Invoker Commands 虽已进入 Baseline 2025，旧设备和未更新浏览器仍可能不支持。
- CSS Anchor Positioning 的部分属性兼容范围并不完全一致，必须逐项查询。
- Scroll-driven Animations 不应制造大幅位移、闪烁或影响内容可读性。
- 新 API 不能替代语义 HTML、键盘测试和网页无障碍检查。
- 上线项目前应基于目标用户浏览器数据决定回退范围。

## 权威参考

- [MDN Invoker Commands API](https://developer.mozilla.org/en-US/docs/Web/API/Invoker_Commands_API)
- [MDN CSS Anchor Positioning](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Anchor_positioning)
- [MDN Scroll-driven Animations](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Scroll-driven_animations)
- [MDN Popover API](https://developer.mozilla.org/en-US/docs/Web/API/Popover_API/Using)
- [网页无障碍基础](../../Basics/1.9-accessibility-basics.md)

本案例为依据 Web 标准和 MDN 文档重新编写的原创教学实现，没有复制外部示例源码。
