
### 1. 标题 (Headings)
使用 `#` 进行标题标记，支持1-6级标题。
```markdown
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```

### 2. 强调 (Emphasis)
#### 斜体 (Italic)
使用 `*` 或 `_` 标记斜体。
```markdown
*斜体文本* 或 _斜体文本_
```
#### 粗体 (Bold)
使用 `**` 或 `__` 标记粗体。
```markdown
**粗体文本** 或 __粗体文本__
```
#### 粗斜体 (Bold Italic)
使用 `***` 或 `___` 同时标记粗体和斜体。
```markdown
***粗斜体文本*** 或 ___粗斜体文本___
```
### 3. 列表 (Lists)
#### 无序列表 (Unordered List)
使用 `-`、`+` 或 `*` 标记无序列表。
```markdown
- 项目1
- 项目2
  - 子项目1
  - 子项目2
```
#### 有序列表 (Ordered List)
使用数字加 `.` 标记有序列表。
```markdown
1. 项目1
2. 项目2
   1. 子项目1
   2. 子项目2
```

### 4. 链接 (Links)
#### 内嵌链接 (Inline Links)
```markdown
[链接文本](https://example.com)
```
#### 引用链接 (Reference Links)
```markdown
这是一个 [引用链接][1] 的例子。

[1]: https://example.com "Example"
```

### 5. 图片 (Images)
```markdown
![替代文本](https://example.com/image.png)
```
你还可以使用引用链接的方式插入图片。
```markdown
![替代文本][image1]

[image1]: https://example.com/image.png "图片标题"
```

### 6. 引用 (Blockquotes)
使用 `>` 表示引用，可以嵌套引用。
```markdown
> 这是一个引用。
>
> > 这是一个嵌套引用。
```

### 7. 代码 (Code)
#### 行内代码 (Inline Code)
使用 `` ` `` 包裹行内代码。
```markdown
这是 `行内代码` 示例。
```
#### 代码块 (Code Blocks)
使用三个反引号 ```` ``` ```` 表示代码块，可以指定语言用于语法高亮。
```markdown
```python
def hello_world():
    print("Hello, World!")
```
```

### 8. 表格 (Tables)
Markdown 支持简单的表格布局，可以对齐表头。
```markdown
| 左对齐 | 居中对齐 | 右对齐 |
|:-------|:--------:|-------:|
| 单元格 | 单元格   | 单元格 |
| 单元格 | 单元格   | 单元格 |
```

### 9. 脚注 (Footnotes)
使用 `[^1]` 标记脚注，并在文末定义。
```markdown
这是一个带有脚注的句子[^1]。

[^1]: 这是脚注的内容。
```

### 10. 嵌套元素 (Nested Elements)
Markdown 支持元素的嵌套，例如列表中的引用或代码块。
```markdown
1. 列表项包含引用：
   > 这是一个引用。
2. 列表项包含代码块：
   ```bash
   echo "Hello, World!"
   ```
```

### 11. 任务列表 (Task Lists)
可以创建任务列表，常用于 TODO 清单。
```markdown
- [x] 已完成的任务
- [ ] 待完成的任务
```

### 12. 换行 (Line Breaks)
在 Markdown 中，行末使用两个或更多空格可以实现换行。
```markdown
这是第一行。  
这是第二行。
```

### 13. 转义字符 (Escaping Characters)
Markdown 中的特殊字符可以使用反斜杠 `\` 进行转义。
```markdown
\*这是被转义的星号\*
```

### 14. 内嵌 HTML (Inline HTML)
Markdown 支持嵌入 HTML，但需注意部分平台可能不支持。
```markdown
<p>This is a paragraph in HTML.</p>
```

### 15. 自动链接 (Automatic Links)
直接输入 URL 或电子邮件地址，会自动转为可点击的链接。
```markdown
https://example.com
```

### 16. 定义列表 (Definition Lists)
虽然不是所有渲染器都支持，但部分 Markdown 扩展支持定义列表。
```markdown
术语
: 定义描述

术语二
: 另一个定义描述
```

### 17. 强调 (Strikethrough)
使用 `~~` 包裹文字来表示删除线。
```markdown
~~这是一个删除线文本。~~
```

### 18. 分割线 (Horizontal Rules)
使用 `---`、`***` 或 `___` 创建分割线。
```markdown
---
```

### 19. 表情符号 (Emoji)
在支持的平台上，可以直接输入表情符号代码，例如 `:smile:`。
```markdown
这是一个笑脸 :smile:
```
