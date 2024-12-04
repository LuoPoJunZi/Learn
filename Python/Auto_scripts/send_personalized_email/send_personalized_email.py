import smtplib  # 导入smtplib模块，用于发送电子邮件
from email.mime.text import MIMEText  # 从email模块导入MIMEText，用于构建电子邮件正文
from email.mime.multipart import MIMEMultipart  # 从email模块导入MIMEMultipart，用于构建多部分邮件

def send_personalized_email(sender_email, sender_password, recipients, subject, body):
    """
    通过Gmail SMTP服务器发送个性化电子邮件给多个收件人。

    参数:
    sender_email (str): 发件人电子邮件地址。
    sender_password (str): 发件人电子邮件密码。
    recipients (list): 收件人电子邮件地址列表。
    subject (str): 电子邮件的主题。
    body (str): 电子邮件的正文内容。
    """
    # 连接到Gmail SMTP服务器，端口为587
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()  # 启用TLS加密
    server.login(sender_email, sender_password)  # 使用发件人电子邮件和密码登录SMTP服务器

    # 遍历每个收件人，发送个性化邮件
    for recipient_email in recipients:
        # 创建MIMEMultipart对象，表示电子邮件
        message = MIMEMultipart()
        message['From'] = sender_email  # 设置发件人地址
        message['To'] = recipient_email  # 设置收件人地址
        message['Subject'] = subject  # 设置邮件主题

        # 将邮件正文内容添加到MIMEMultipart对象中
        message.attach(MIMEText(body, 'plain'))

        # 发送邮件给当前收件人
        server.send_message(message)

    # 发送完成后，断开与SMTP服务器的连接
    server.quit()

# 使用示例
if __name__ == "__main__":
    # 定义发件人信息、收件人列表、邮件主题和正文
    sender_email = 'your_email@gmail.com'  # 发件人电子邮件地址
    sender_password = 'your_password'  # 发件人电子邮件密码
    recipients = ['recipient1@example.com', 'recipient2@example.com']  # 收件人电子邮件地址列表
    subject = 'Hello'  # 邮件主题
    body = 'This is a test email.'  # 邮件正文

    # 调用send_personalized_email函数发送邮件
    send_personalized_email(sender_email, sender_password, recipients, subject, body)
