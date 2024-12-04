import sqlite3  # 导入 sqlite3 库，用于与 SQLite 数据库进行交互

# 连接到 SQLite 数据库的函数
def connect_to_database(db_path):
    """
    连接到指定路径的 SQLite 数据库，并返回连接对象和游标。

    参数:
    db_path (str): 数据库文件的路径。

    返回:
    tuple: 包含连接对象和游标的元组。
    """
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_path)

    # 创建游标对象，用于执行 SQL 查询
    cursor = conn.cursor()

    return conn, cursor  # 返回连接对象和游标

# 执行 SQL 查询的函数
def execute_query(cursor, query):
    """
    使用游标执行指定的 SQL 查询，并返回查询结果。

    参数:
    cursor (sqlite3.Cursor): SQLite 数据库的游标。
    query (str): 要执行的 SQL 查询语句。

    返回:
    list: 查询结果，以列表的形式返回。
    """
    # 执行 SQL 查询
    cursor.execute(query)

    # 获取所有查询结果
    results = cursor.fetchall()

    return results  # 返回查询结果

# 使用示例
if __name__ == "__main__":
    # 连接到数据库
    conn, cursor = connect_to_database('/path/to/database.db')  # 请将 '/path/to/database.db' 替换为实际数据库路径

    # 执行查询以获取表中的所有记录
    query = 'SELECT * FROM table_name'  # 替换 'table_name' 为实际的表名
    results = execute_query(cursor, query)

    # 打印查询结果
    print(results)

    # 关闭数据库连接
    conn.close()
