### **脚本功能说明**

这个Python脚本的主要功能是**连接到SQLite数据库并执行SQL查询**。它使用 `sqlite3` 库来连接到数据库，创建游标对象，用于执行查询并返回查询结果。具体步骤如下：

1. **连接到数据库**：指定数据库路径，连接到SQLite数据库，并创建一个游标对象。
2. **执行SQL查询**：使用游标执行SQL查询，并返回查询结果。
3. **关闭连接**：执行完查询后，关闭数据库连接以释放资源。

### **带注释的Python脚本**

```python
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
```

### **代码详解**

1. **导入必要的模块**
    ```python
    import sqlite3
    ```
    - `sqlite3` 是Python内置的模块，用于与SQLite数据库进行交互。可以用于连接数据库、执行SQL语句以及管理数据库事务。

2. **定义 `connect_to_database` 函数**
    ```python
    def connect_to_database(db_path):
        """
        连接到指定路径的 SQLite 数据库，并返回连接对象和游标。
        """
        conn = sqlite3.connect(db_path)  # 连接到 SQLite 数据库
        cursor = conn.cursor()  # 创建游标对象
        return conn, cursor
    ```
    - **参数**：
        - `db_path`：数据库文件的路径（字符串）。
    - **功能**：
        - 使用 `sqlite3.connect(db_path)` 连接到指定的SQLite数据库文件。
        - 使用 `conn.cursor()` 创建游标对象，游标用于执行SQL查询。
        - 返回连接对象 `conn` 和游标 `cursor`。

3. **定义 `execute_query` 函数**
    ```python
    def execute_query(cursor, query):
        """
        使用游标执行指定的 SQL 查询，并返回查询结果。
        """
        cursor.execute(query)  # 执行 SQL 查询
        results = cursor.fetchall()  # 获取所有查询结果
        return results
    ```
    - **参数**：
        - `cursor`：SQLite数据库的游标对象。
        - `query`：SQL查询语句（字符串）。
    - **功能**：
        - 使用 `cursor.execute(query)` 执行指定的SQL查询。
        - 使用 `cursor.fetchall()` 获取查询结果，并将其返回。
        - 查询结果是以列表形式返回，每一项是查询结果中的一行。

4. **使用示例**
    ```python
    if __name__ == "__main__":
        conn, cursor = connect_to_database('/path/to/database.db')
        query = 'SELECT * FROM table_name'
        results = execute_query(cursor, query)
        print(results)
        conn.close()
    ```
    - 这部分代码确保当脚本作为主程序运行时执行。
    - 调用 `connect_to_database` 函数，连接到 `/path/to/database.db` 数据库，并创建游标。
    - 调用 `execute_query` 函数，执行SQL查询 `SELECT * FROM table_name`，获取查询结果。
    - 打印查询结果并关闭数据库连接。

### **使用示例**

假设您有一个SQLite数据库文件 `database.db`，其中有一个表 `employees`。运行这个脚本后，查询该表中的所有记录，并输出查询结果。

### **注意事项**

1. **数据库路径**
    - 在调用 `connect_to_database` 函数时，需要提供实际存在的SQLite数据库文件的路径。请确保文件路径正确，数据库文件能够被脚本访问。

2. **SQL查询的正确性**
    - 在调用 `execute_query` 函数时，确保SQL查询语句是正确的。例如，表名需要与数据库中的实际表名匹配，否则会引发错误。

3. **游标与连接的使用**
    - 连接对象 `conn` 和游标对象 `cursor` 是相互关联的。完成所有数据库操作后，应该调用 `conn.close()` 以关闭连接，从而释放数据库资源。

4. **异常处理**
    - 脚本中没有添加异常处理，如果连接失败或者查询有错误，可能会导致脚本崩溃。可以使用 `try-except` 结构来捕获异常，并适当处理。
    ```python
    def connect_to_database(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            return conn, cursor
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return None, None
    ```

### **扩展功能建议**

1. **执行其他类型的SQL语句**
    - 可以扩展脚本以执行插入、更新和删除操作。不同于查询操作，这些操作需要对数据库进行修改，因此执行后需要调用 `conn.commit()` 来保存更改。
    ```python
    def execute_non_query(cursor, conn, query):
        """
        使用游标执行指定的非查询 SQL 语句（如插入、更新或删除）。

        参数:
        cursor (sqlite3.Cursor): SQLite 数据库的游标。
        conn (sqlite3.Connection): SQLite 数据库的连接对象。
        query (str): 要执行的 SQL 语句。

        返回:
        None
        """
        try:
            cursor.execute(query)
            conn.commit()  # 提交事务，保存更改
            print("Query executed successfully.")
        except sqlite3.Error as e:
            print(f"Error executing query: {e}")

    # 使用示例
    insert_query = "INSERT INTO table_name (column1, column2) VALUES ('value1', 'value2')"
    execute_non_query(cursor, conn, insert_query)
    ```

2. **自动关闭连接的上下文管理器**
    - 可以将数据库连接封装到上下文管理器中，以确保使用完成后自动关闭连接。
    ```python
    from contextlib import contextmanager

    @contextmanager
    def database_connection(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            conn.close()

    # 使用示例
    with database_connection('/path/to/database.db') as cursor:
        query = 'SELECT * FROM table_name'
        cursor.execute(query)
        results = cursor.fetchall()
        print(results)
    ```

3. **动态传递参数的查询**
    - 可以扩展查询函数以支持动态传递参数，使用占位符来防止SQL注入攻击。
    ```python
    def execute_query_with_params(cursor, query, params):
        """
        使用游标执行带参数的 SQL 查询，并返回查询结果。

        参数:
        cursor (sqlite3.Cursor): SQLite 数据库的游标。
        query (str): 要执行的 SQL 查询语句。
        params (tuple): 查询参数。

        返回:
        list: 查询结果，以列表的形式返回。
        """
        cursor.execute(query, params)
        results = cursor.fetchall()
        return results

    # 使用示例
    param_query = 'SELECT * FROM table_name WHERE column1 = ?'
    results = execute_query_with_params(cursor, param_query, ('value1',))
    print(results)
    ```

4. **批量执行SQL语句**
    - 可以扩展脚本以批量执行多个SQL语句，适合需要一次性插入或更新大量数据的场景。
    ```python
    def execute_many(cursor, query, data):
        """
        批量执行相同的 SQL 语句。

        参数:
        cursor (sqlite3.Cursor): SQLite 数据库的游标。
        query (str): 要执行的 SQL 语句。
        data (list): 包含每次执行时使用的参数的列表。

        返回:
        None
        """
        cursor.executemany(query, data)
        conn.commit()  # 提交事务，保存更改

    # 使用示例
    batch_query = "INSERT INTO table_name (column1, column2) VALUES (?, ?)"
    batch_data = [('value1', 'value2'), ('value3', 'value4')]
    execute_many(cursor, batch_query, batch_data)
    ```

### **总结**

这个脚本是一个简单的SQLite数据库操作工具，用于连接到数据库并执行SQL查询。它利用Python内置的 `sqlite3` 模块，提供了连接数据库、执行SQL查询和关闭连接的基本功能。通过 `connect_to_database` 和 `execute_query` 函数，用户可以轻松地连接数据库并执行查询。

在扩展功能方面，可以添加支持执行插入、更新、删除等修改数据库内容的SQL语句，同时添加批量操作、参数化查询和上下文管理器来确保资源释放。此外，增加异常处理可以使脚本更健壮，在出现连接失败或SQL错误时提供有意义的错误信息和日志记录。
