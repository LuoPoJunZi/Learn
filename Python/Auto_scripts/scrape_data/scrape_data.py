import requests  # 导入requests库，用于发送HTTP请求
from bs4 import BeautifulSoup  # 从bs4库导入BeautifulSoup，用于解析HTML

def scrape_data(url):
    """
    从指定的URL抓取网页内容并解析HTML。

    参数:
    url (str): 目标网页的URL地址。

    返回:
    BeautifulSoup对象: 解析后的HTML内容。
    """
    try:
        # 发送HTTP GET请求到指定的URL
        response = requests.get(url)
        # 检查请求是否成功（状态码200）
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # 如果请求过程中发生异常，打印错误信息并返回None
        print(f"请求失败: {e}")
        return None

    # 使用BeautifulSoup解析获取到的HTML文本，使用'html.parser'作为解析器
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 在此处添加从网站提取相关数据的代码
    # 例如，可以提取所有的段落文本：
    # paragraphs = soup.find_all('p')
    # for p in paragraphs:
    #     print(p.get_text())

    return soup  # 返回解析后的BeautifulSoup对象

# 使用示例
if __name__ == "__main__":
    url = 'https://example.com'  # 定义要抓取的目标URL
    soup = scrape_data(url)  # 调用scrape_data函数获取解析后的HTML内容
    if soup:
        # 打印网页的标题
        print(soup.title.string)
