# Markdown 学习导读

Markdown 的核心价值不是“写得花”，而是让别人容易读、容易改、容易维护。学 Markdown 最重要的是建立文档结构感。

## 适合人群

- 想写 GitHub README、教程、博客、学习笔记的人。
- 已经会基础语法，但文档看起来不够清楚的人。
- 想让文档在 GitHub、博客、编辑器之间尽量兼容的人。

## 学习目标

学完后，你应该能：

- 写出结构清晰的 README。
- 正确使用标题、列表、表格、代码块和链接。
- 区分通用 Markdown 与平台扩展语法。
- 用 Mermaid 表达简单流程。

## 推荐阅读顺序

1. [Markdown 用法教程](README.md)
2. [Markdown 示例与模板](examples/README.md)
3. [README 模板](examples/readme-template.md)
4. 本文的 README 写作建议和平台差异说明

## README 写作结构

一个新手友好的 README 可以按这个顺序写：

```text
项目名称
-> 这个项目解决什么问题
-> 适合谁使用
-> 目录或功能说明
-> 如何安装或打开
-> 如何运行或修改
-> 常见问题
-> 来源、许可证或注意事项
```

如果项目只是教程，重点写清“先读什么、后读什么、遇到问题去哪查”。如果项目是脚本，重点写清“输入、输出、依赖、风险”。

## GitHub README 美化原则

推荐做：

- 使用一张清晰的目录表。
- 给命令标注语言，例如 `powershell`、`bash`、`python`。
- 用表格整理入口、用途、适合人群。
- 给图片写替代文本。
- 外部资源说明用途，不只贴链接。

谨慎做：

- 过多徽章。
- 过宽表格。
- 大量 HTML 混写。
- 只在某个平台生效的主题标签。

## 博客文章结构

博客更适合讲“为什么”和“踩坑过程”，可以按这个结构：

```text
问题背景
-> 尝试过程
-> 最终方案
-> 关键命令或代码
-> 常见错误
-> 总结
```

教程类博客不要只贴最终代码。新手真正需要的是：为什么这样写、每一步在哪里验证。

## Mermaid 入门

Mermaid 适合画流程图、时序图、状态图。最常用的是流程图：

````markdown
```mermaid
flowchart LR
    A["写文档"] --> B["检查链接"]
    B --> C["提交到 GitHub"]
```
````

写 Mermaid 时，节点文字里有中文、空格或标点，建议用引号包起来。

## 平台差异

| 语法 | GitHub | 常见博客 | 说明 |
| :--- | :--- | :--- | :--- |
| 标题、列表、链接 | 支持 | 通常支持 | 最稳定，优先使用。 |
| 表格 | 支持 | 通常支持 | 表格太宽会影响移动端阅读。 |
| 任务列表 | 支持 | 不一定 | GitHub 项目文档很好用。 |
| 数学公式 | 支持情况变化 | 常见支持 | 发布前要预览。 |
| Mermaid | GitHub 支持 | 不一定 | 适合流程说明。 |
| 主题标签插件 | 不支持 | 取决于主题 | 不适合通用 README。 |

## 少量自查

- 你的 README 第一屏能看出项目用途吗？
- 每个本地链接都能从当前文件点开吗？
- 代码块有没有标注语言？
- 平台专属语法是否写了说明？

## 外部资源

- [CommonMark](https://commonmark.org/)：Markdown 标准化规范。
- [GitHub Flavored Markdown Spec](https://github.github.com/gfm/)：GitHub 使用的 Markdown 扩展规范。
- [Markdown Guide](https://www.markdownguide.org/)：适合查语法和平台支持情况。
- [Mermaid Docs](https://mermaid.js.org/)：Mermaid 图表官方文档。

## 下一步

建议拿一个你自己的项目，先用 [README 模板](examples/readme-template.md) 写出骨架，再逐步补充截图、运行说明和常见问题。
