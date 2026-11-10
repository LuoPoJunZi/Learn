# DuckDB 数据分析入门：从 CSV 汇总到 Parquet

## 适合人群与学习目标

适合会运行 Python、读过文件读写教程，但还不熟悉 SQL 的学习者。目标是用一个订单表理解筛选、分组与聚合，区分 CSV 和 Parquet，并学会在本地内存数据库中查询数据。

官方资料核验：**2026-11-10**。本页参考 DuckDB 2.0 文档；该文档要求 Python 3.11+，不支持自由线程 CPython 构建。DuckDB 是成熟的嵌入式分析数据库，不是这周刚发明的新技术。当前工作环境未安装 DuckDB，本文不冒充已完成真实 DuckDB 执行验证。

## 一、先认识三个工具

| 名称 | 解决什么问题 | 本例用途 |
| :--- | :--- | :--- |
| CSV | 用分隔文本存表格，方便交换 | 输入五条订单数据 |
| DuckDB | 在进程内执行分析型 SQL，不必启动数据库服务 | 筛选、汇总、排序 |
| Parquet | 带类型、按列组织的数据文件格式 | 可选的临时文件往返验证 |

DuckDB 与 pandas 可以互补，不是谁必须取代谁。按列分析时 Parquet 有优势，但不意味着每个小文件都比 CSV 更小、更快；测试还受数据分布、压缩方式和读取列数影响。

## 文件入口与安全默认值

- [sales.csv](sales.csv)：五条固定订单，金额使用整数“分”。
- [duckdb_sales.py](duckdb_sales.py)：读取固定本地 CSV，在 `:memory:` 连接中执行查询。
- [回归测试](../../scripts/tests/test_learning_workflows.py)：有 DuckDB 时核对汇总、筛选、参数拒绝与 Parquet 往返；缺依赖时明确跳过。

默认运行只读取示例 CSV、打印结果，不写输出文件。添加 `--parquet` 时仅在新建的系统临时目录内写一个 Parquet 文件，退出后清理临时目录，不修改原 CSV 或用户数据。分析引擎资源使用仍受运行环境影响；真实大数据任务需另行设置内存和临时存储预算。

## 二、先在独立环境安装依赖

以下步骤由读者手动执行，会创建环境，安装依赖时可能联网。本轮没有执行这些安装命令，也不向仓库根目录加入全局依赖。

在仓库根目录、Windows PowerShell 中：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "duckdb>=2.0,<2.1"
.\.venv\Scripts\python.exe -c "import duckdb; print(duckdb.__version__)"
```

若已有 `.venv`，先确认解释器与用途，不要覆盖重要环境。Linux/macOS 对应使用 `python3 -m venv .venv` 与 `.venv/bin/python`。上述版本范围用于跟随本教程的 2.0 文档，不意味着任何时间的最新版本都必须如此；跨大版本升级要重跑测试。

## 三、从五行订单看懂 SQL

输入字段：订单 ID、类别、数量、单价（分）。本例显式指定字段类型，避免 CSV 自动推断随着输入内容变化而改变。

核心查询如下，源码通过参数绑定传入最小数量：

```sql
SELECT category,
       SUM(quantity) AS units,
       SUM(quantity * unit_price_cents) AS revenue_cents
FROM sales
WHERE quantity >= ?
GROUP BY category
ORDER BY revenue_cents DESC, category;
```

逐行理解：

1. `FROM sales`：读取已注册的本地数据视图。
2. `WHERE quantity >= ?`：先过滤订单行，不是先求和再过滤类别。
3. `GROUP BY category`：把相同类别放入一组。
4. `SUM`：分别累计数量与金额。
5. `ORDER BY`：金额降序；金额相同时按类别名排序，保证稳定输出。

金额使用整数分避免把本例变成浮点精度演示；复杂财务规则可使用固定精度 DECIMAL，并明确舍入与币种，不直接用二进制 float 计算账单。

## 四、运行并核对结果

在装有 DuckDB 的环境中，从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe "Python/Data Analysis/duckdb_sales.py"
```

按输入数据手算得到的预期结果为：

| 类别 | 总数量 | 总金额（分） |
| :--- | ---: | ---: |
| courses | 3 | 19700 |
| books | 3 | 11500 |
| tools | 3 | 3600 |

这些是固定输入对应的预期值，不是本轮真实 DuckDB 运行报告。例如课程类金额为 `1×9900 + 2×4900 = 19700`。

`summary(connection, 2)` 只保留单行数量至少为 2 的订单；它不会根据“类别总数量至少为 2”筛选。后者应使用 `HAVING SUM(quantity) >= ?`，不要混淆 `WHERE` 与 `HAVING`。

## 五、体验 Parquet，不覆盖源文件

确认允许写入临时测试目录后，再手动运行：

```powershell
.\.venv\Scripts\python.exe "Python/Data Analysis/duckdb_sales.py" --parquet
```

流程是：CSV 视图 → 写临时 Parquet → 重新读取 → 使用同一 SQL 汇总 → 与原结果逐项比较。文件大小只是本次文件的观测值，不应据此对五行数据做性能宣传。

Parquet 的列式布局有利于只读取需要的列；统计信息也可能帮助跳过不相关的数据。但具体是否发生投影或过滤下推，应结合查询计划和文件布局验证，不能只看到扩展名就断言一定提速。

## 六、为什么绑定参数

源码使用：

```python
connection.execute(SUMMARY_SQL, [min_quantity]).fetchall()
```

`?` 用于绑定值，避免将用户输入拼接成 SQL。它不代表可以把任意表名、列名或 SQL 片段作为值参数替换；动态标识符需要受控白名单或正规接口。

本例只读取代码目录内固定的 `sales.csv`，不接受网络 URL、任意路径或通配符。不将不可信 SQL 暴露成服务，也不自动安装扩展。DuckDB 能读写文件和加载扩展，不能把“在进程内执行”理解为天然安全沙箱。

## 测试、常见错误与完成检查

在相同环境执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s scripts/tests -p "test_learning_workflows.py" -v
```

安装 DuckDB 后，三项 DuckDB 测试应实际运行；未安装时显示 `skipped`，不算验证了 SQL 或 Parquet。

- **报 DuckDB 未安装**：检查运行脚本的解释器是否与安装依赖时相同。
- **从别的工作目录运行找不到 CSV**：本例依据 `__file__` 定位，不依赖当前工作目录；自己改代码时保持这一点。
- **数字被当成字符串**：核对字段类型与坏行，别通过忽略错误悄悄丢数据。
- **排序每次不同**：没有 `ORDER BY` 的查询不应依赖默认返回顺序。
- **查询没输出**：关系对象可能惰性执行，要在需要时取结果或展示；本例明确调用 `fetchall()`。

完成时应能手算三类金额、解释过滤发生在哪一步、区分 CSV 与 Parquet，并核对三项测试不是被跳过。

## 官方来源与下一步

- [DuckDB 2.0 Python API](https://duckdb.org/docs/2.0/clients/python/overview.html)：版本要求、连接与关系 API。
- [DuckDB 2.0 数据读取](https://duckdb.org/docs/2.0/clients/python/data_ingestion.html)：CSV/Parquet 与显式读取方式。
- [Parquet 读写](https://duckdb.org/docs/2.0/data/parquet/overview.html)：列式文件、查询和导出。
- [Python 环境指南](../ENVIRONMENT_GUIDE.md)、[自动化脚本索引](../Auto_scripts/README.md)：接回已有环境与表格处理路线。

代码和样本为原创教学实现，不含用户订单或真实客户数据。本文没有运行依赖安装、联网查询或对实际业务目录执行操作。
