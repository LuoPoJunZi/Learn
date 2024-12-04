import requests  # 导入 requests 库，用于发送 HTTP 请求

# 检查网站状态的函数
def check_website_status(url):
    """
    检查指定网站的状态，并输出相应信息。

    参数:
    url (str): 要检查的网站的 URL。
    """
    try:
        # 发送 GET 请求到指定 URL
        response = requests.get(url)

        # 检查响应的状态码
        if response.status_code == 200:
            print(f"Website {url} is up and running.")  # 如果状态码为 200，表示网站正常
        else:
            print(f"Website {url} returned status code {response.status_code}.")  # 输出非 200 状态码

    # 捕获请求过程中发生的异常
    except requests.exceptions.RequestException as e:
        print(f"Error accessing website {url}: {e}")  # 打印错误信息

# 使用示例
if __name__ == "__main__":
    # 调用 check_website_status 函数，检查指定网站的状态
    check_website_status('https://example.com')
