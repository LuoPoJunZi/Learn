# Love 表白网页示例

这个目录保存几个静态表白网页示例，适合用来学习 HTML 结构、CSS 样式、JavaScript 动画和简单页面改造。

这些页面主要用于学习和个人展示练习。修改时建议先复制一份目录做实验，保留原始示例方便对照。

## 示例列表

| 示例 | 入口 | 说明 |
| :--- | :--- | :--- |
| LoveYou-15 | [LoveYou-15/index.html](LoveYou-15/index.html) | 包含独立 JavaScript 文件，适合学习脚本拆分 |
| LoveYou-32 | [LoveYou-32/index.html](LoveYou-32/index.html) | 单页示例，适合练习直接修改页面内容 |
| LoveYou-33 | [LoveYou-33/index.html](LoveYou-33/index.html) | 单页示例，适合练习样式和动画调整 |

## 如何查看

这些都是静态页面，通常可以直接双击对应目录下的 `index.html` 文件，用浏览器打开。

如果你想用 VS Code 查看和修改：

1. 用 VS Code 打开整个 `HTML` 目录。
2. 找到对应示例的 `index.html`。
3. 修改文字、样式或脚本。
4. 回到浏览器刷新页面查看效果。

如果页面显示异常，先确认：

- 目录结构没有被打乱。
- `index.html` 旁边的 `js`、图片或其他资源文件还在原位置。
- 浏览器控制台是否有红色报错。

## 推荐修改顺序

新手建议按这个顺序改：

1. 修改标题、按钮文字、页面文案。
2. 修改颜色、字体大小、背景色。
3. 替换图片或背景资源。
4. 调整动画速度、延迟时间。
5. 最后再改 JavaScript 逻辑。

每次只改一小处，然后刷新页面确认效果。这样比较容易定位问题。

## LoveYou-15 的脚本位置

`LoveYou-15` 的 JavaScript 单独放在：

- [LoveYou-15/js/index.js](LoveYou-15/js/index.js)

如果你想修改动画、点击效果、计时逻辑，优先查看这个文件。

## 常见问题

### 修改后页面没有变化

可能是浏览器缓存。可以尝试按 `Ctrl + F5` 强制刷新，或者关闭页面后重新打开。

### 中文乱码

确认 HTML 文件中包含类似下面的编码声明：

```html
<meta charset="UTF-8">
```

同时确认编辑器保存编码是 UTF-8。

### 图片不显示

检查图片路径是否正确。HTML 里的路径是相对 `index.html` 所在目录计算的，不是相对仓库根目录。

## 学习建议

如果你还不熟悉网页基础，建议先读：

- [HTML 基础教程](../Basics/README.md)
- [网页结构](../Basics/1.1-webpage-structure.md)
- [CSS 基础](../Basics/1.5-css-basics.md)
- [JavaScript 基础](../Basics/1.7-javascript-basics.md)

## 来源说明

部分示例来自公开项目或开源代码，并经过整理注释。更多类似示例可参考 [Awesome-Love-Code](https://github.com/sun0225SUN/Awesome-Love-Code)。
