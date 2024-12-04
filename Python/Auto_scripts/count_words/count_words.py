def count_words(file_path):
    """
    统计指定文件中的单词总数。

    参数:
    file_path (str): 文件的路径。

    返回:
    int: 文件中的单词总数。
    """
    # 使用 'with' 语句打开文件，确保文件使用后自动关闭
    with open(file_path, 'r') as f:
        text = f.read()  # 读取文件的全部内容

    # 使用 split() 方法将文本按空格拆分为单词列表
    word_count = len(text.split())

    # 返回单词总数
    return word_count

# 使用示例
if __name__ == "__main__":
    # 调用 count_words 函数，传入文件路径并获取单词总数
    word_count = count_words('/path/to/file.txt')  # 请将 '/path/to/file.txt' 替换为实际文件路径
    # 打印单词总数
    print(f"Word count: {word_count}")
