import pandas as pd  # 导入 pandas 库，用于处理数据

# 从Excel文件中移除重复行的函数
def remove_duplicates(file_path):
    """
    从指定Excel文件中移除重复行，并将结果保存回原文件。

    参数:
    file_path (str): Excel文件的路径。
    """
    # 使用 pandas 读取 Excel 文件，存储为 DataFrame
    df = pd.read_excel(file_path)

    # 使用 drop_duplicates() 方法移除重复的行
    df.drop_duplicates(inplace=True)

    # 将修改后的 DataFrame 写入原 Excel 文件中，覆盖原文件
    df.to_excel(file_path, index=False)  # index=False 表示不保存行索引

# 使用示例
if __name__ == "__main__":
    # 调用 remove_duplicates 函数，移除指定 Excel 文件中的重复行
    remove_duplicates('/path/to/data.xlsx')  # 请将 '/path/to/data.xlsx' 替换为实际文件路径
