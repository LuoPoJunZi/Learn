import tweepy  # 导入 tweepy 库，用于与 Twitter API 进行交互

# 发布推文的函数
def post_tweet(api_key, api_secret, access_token, access_token_secret, message):
    """
    使用提供的凭据通过 Twitter API 发布推文。

    参数:
    api_key (str): Twitter API 的 API Key。
    api_secret (str): Twitter API 的 API Secret Key。
    access_token (str): 访问令牌（Access Token）。
    access_token_secret (str): 访问令牌密钥（Access Token Secret）。
    message (str): 要发布的推文内容。
    """
    # 使用 API Key 和 API Secret Key 进行 OAuth 认证
    auth = tweepy.OAuthHandler(api_key, api_secret)

    # 设置访问令牌和访问令牌密钥
    auth.set_access_token(access_token, access_token_secret)

    # 创建 Twitter API 对象
    api = tweepy.API(auth)

    # 使用 API 对象发布推文
    api.update_status(message)

# 使用示例
if __name__ == "__main__":
    # Twitter 开发者凭据
    api_key = 'your_api_key'  # 请替换为您的 Twitter API Key
    api_secret = 'your_api_secret'  # 请替换为您的 Twitter API Secret Key
    access_token = 'your_access_token'  # 请替换为您的访问令牌
    access_token_secret = 'your_access_token_secret'  # 请替换为您的访问令牌密钥

    # 要发布的推文内容
    message = 'Hello, Twitter!'

    # 调用 post_tweet 函数发布推文
    post_tweet(api_key, api_secret, access_token, access_token_secret, message)
