import pandas as pd  # 导入pandas库，用于处理Excel文件

# 读取Excel文件的函数
def read_excel(file_path):
    """
    从指定路径读取Excel文件，并返回数据框。

    参数:
    file_path (str): Excel文件的路径。

    返回:
    DataFrame: 从Excel文件中读取的数据。
    """
    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(file_path)
    return df  # 返回读取到的数据框

# 将数据写入Excel文件的函数
def write_to_excel(data, file_path):
    """
    将数据写入Excel文件。

    参数:
    data (dict or DataFrame): 要写入Excel的数据，可以是字典或DataFrame。
    file_path (str): 输出Excel文件的路径。
    """
    # 将数据转换为DataFrame对象
    df = pd.DataFrame(data)
    # 使用 pandas 将 DataFrame 写入 Excel 文件
    df.to_excel(file_path, index=False)  # index=False 表示不写入行索引

# 使用示例
if __name__ == "__main__":
    # 定义数据，字典形式
    data = {'Column1': [1, 2, 3], 'Column2': [4, 5, 6]}
    # 调用 write_to_excel 函数，将数据写入 Excel 文件
    write_to_excel(data, '/path/to/output.xlsx')  # 请将 '/path/to/output.xlsx' 替换为实际路径

    # 调用 read_excel 函数，从 Excel 文件读取数据
    df = read_excel('/path/to/output.xlsx')  # 请将 '/path/to/output.xlsx' 替换为实际路径
    # 打印读取到的数据框
    print(df)
