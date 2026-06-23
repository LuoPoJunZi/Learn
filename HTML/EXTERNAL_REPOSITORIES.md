# HTML 优秀开源项目导读

这份文档给初学者整理一些值得参考的 HTML、CSS、JavaScript 开源项目和学习资源。它们适合用来找灵感、拆页面结构、练习交互组件，但不建议不加理解地整包复制。

## 推荐资源总览

| 资源 | 适合阶段 | 主要用途 | 许可证或注意事项 |
| :--- | :--- | :--- | :--- |
| [bradtraversy/50projects50days](https://github.com/bradtraversy/50projects50days) | 入门到进阶 | 练习 50+ 个 HTML/CSS/JS 小项目 | MIT License |
| [StartBootstrap Landing Page](https://github.com/StartBootstrap/startbootstrap-landing-page) | 入门后 | 学习 Bootstrap 落地页结构 | MIT License |
| [StartBootstrap Creative](https://github.com/StartBootstrap/startbootstrap-creative) | 入门后 | 学习单页创意网站结构 | MIT License |
| [HTML5 UP](https://html5up.net/) | 入门后 | 参考高质量响应式模板 | CC BY 3.0，需要署名 |
| [h5bp/html5-boilerplate](https://github.com/h5bp/html5-boilerplate) | 进阶 | 学习专业 HTML5 项目起步结构 | MIT License |
| [learning-zone/website-templates](https://github.com/learning-zone/website-templates) | 入门后 | 查找大量模板灵感 | 使用前检查具体模板来源和许可证 |
| [freeCodeCamp Responsive Web Design](https://www.freecodecamp.org/learn/2022/responsive-web-design/) | 零基础到入门 | 系统学习 HTML/CSS | 在线课程 |

## 怎么使用这些资源

推荐顺序：

1. 先学习本仓库 [HTML 基础教程](Basics/README.md)。
2. 再看 [HTML 案例集合](Examples/README.md)，先改本地轻量案例。
3. 然后去外部项目中挑一个小页面或小组件。
4. 先运行，再拆结构，最后自己仿写一版。

不要一开始就下载大型模板硬改。对新手来说，更有效的方式是：

```text
看页面效果 -> 找 HTML 结构 -> 找 CSS 布局 -> 找 JS 交互 -> 自己改一个小功能
```

## bradtraversy/50projects50days

仓库地址：

- [https://github.com/bradtraversy/50projects50days](https://github.com/bradtraversy/50projects50days)

这个仓库包含 50+ 个 HTML/CSS/JavaScript 小项目，例如扩展卡片、进度步骤、隐藏搜索框、滚动动画、FAQ 折叠、Toast 通知、Todo List 等。

适合学习：

- 小型交互组件
- CSS 动画
- DOM 操作
- 事件监听
- 单页面小项目拆分方式

建议挑这些先看：

- Expanding Cards
- Progress Steps
- Hidden Search Widget
- Faq Collapse
- Toast Notification
- Todo List

## StartBootstrap

资源地址：

- [StartBootstrap Landing Page](https://github.com/StartBootstrap/startbootstrap-landing-page)
- [StartBootstrap Creative](https://github.com/StartBootstrap/startbootstrap-creative)

StartBootstrap 提供很多 Bootstrap 模板。Landing Page 适合学习产品落地页，Creative 适合学习单页品牌展示。

适合学习：

- 首屏 Hero 区域
- 响应式导航
- Bootstrap 栅格布局
- 页面分区组织
- CTA 按钮和表单区域

注意：

- 这些模板通常包含构建工具和源文件目录。
- 新手优先看 `dist` 或编译后的 HTML/CSS/JS。
- 不要一开始就陷入 npm、Pug、SCSS 的细节。

## HTML5 UP

网站地址：

- [https://html5up.net/](https://html5up.net/)

HTML5 UP 提供很多设计成熟的响应式模板。它们适合观察布局、留白、字体层级、卡片和响应式断点。

注意：

- HTML5 UP 模板使用 CC BY 3.0 许可证。
- 可以免费使用和修改，但需要保留署名。
- 如果要放进自己的项目，先阅读其许可证说明。

## h5bp/html5-boilerplate

仓库地址：

- [https://github.com/h5bp/html5-boilerplate](https://github.com/h5bp/html5-boilerplate)

HTML5 Boilerplate 是专业前端项目的起步模板。它不是“好看的页面模板”，更像是一套稳健的网页工程基础。

适合学习：

- 标准 HTML5 页面结构
- 跨浏览器兼容思路
- 默认 CSS 组织
- 项目起步文件结构
- 静态资源管理

新手建议先读结构，不要急着全部照搬。

## learning-zone/website-templates

仓库地址：

- [https://github.com/learning-zone/website-templates](https://github.com/learning-zone/website-templates)

这个仓库收集了大量 HTML5 模板，适合按行业或页面类型找灵感。

适合查找：

- Portfolio
- Restaurant
- Business
- Education
- App Landing Page
- Coming Soon
- Admin Template

注意：

- 模板来源较多，使用前要逐个检查许可证和资源来源。
- 不建议直接整包复制到自己的项目。
- 可以选择一个页面，学习它的布局方式和配色思路。

## freeCodeCamp Responsive Web Design

课程地址：

- [https://www.freecodecamp.org/learn/2022/responsive-web-design/](https://www.freecodecamp.org/learn/2022/responsive-web-design/)

这是一条系统的 HTML/CSS 学习路线，适合零基础循序渐进练习。

适合学习：

- HTML 语义标签
- CSS 选择器
- Flexbox
- Grid
- 响应式设计
- 表单和可访问性基础

## 初学者安全提醒

- 不要直接运行来路不明的压缩脚本。
- 不要把外部模板中的追踪代码、广告代码、统计代码直接复制进自己的页面。
- 不要随便提交大体积图片、字体、视频素材。
- 使用外部模板前先看 LICENSE。
- 如果模板要求署名，要保留原作者信息。

