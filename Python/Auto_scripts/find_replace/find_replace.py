def find_replace(file_path, search_text, replace_text):
    """
    在指定文件中查找并替换文本。

    参数:
    file_path (str): 要操作的文件路径。
    search_text (str): 要查找的文本。
    replace_text (str): 要替换为的新文本。
    """
    # 使用 'with' 语句打开文件，读取文件内容
    with open(file_path, 'r') as f:
        text = f.read()  # 读取整个文件内容为字符串

    # 使用 replace() 方法将所有匹配的文本替换为新的文本
    modified_text = text.replace(search_text, replace_text)

    # 再次使用 'with' 语句，以写入模式打开文件，将修改后的内容写入文件，覆盖原内容
    with open(file_path, 'w') as f:
        f.write(modified_text)  # 写入修改后的内容

# 使用示例
if __name__ == "__main__":
    # 调用 find_replace 函数，传入文件路径、要查找的文本和替换的文本
    find_replace('/path/to/file.txt', 'old', 'new')  # 请将 '/path/to/file.txt' 替换为实际文件路径
