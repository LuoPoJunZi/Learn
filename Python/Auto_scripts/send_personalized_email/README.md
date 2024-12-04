### **脚本功能说明**

这个Python脚本的主要功能是**使用SMTP协议通过Gmail服务器向多个收件人发送个性化电子邮件**。具体步骤如下：

1. 连接到Gmail SMTP服务器并进行身份验证。
2. 创建电子邮件消息并为每个收件人添加邮件内容。
3. 通过SMTP服务器发送邮件。
4. 完成所有发送操作后，断开与SMTP服务器的连接。

### **带注释的Python脚本**

```python
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
```

### **代码详解**

1. **导入必要的模块**
    ```python
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    ```
    - `smtplib`：用于与SMTP服务器进行交互，发送电子邮件。
    - `MIMEText`：用于创建电子邮件正文内容。
    - `MIMEMultipart`：用于创建包含多个部分（如正文和附件）的电子邮件。

2. **定义 `send_personalized_email` 函数**
    ```python
    def send_personalized_email(sender_email, sender_password, recipients, subject, body):
        ...
    ```
    - **参数**：
        - `sender_email`：发件人的电子邮件地址。
        - `sender_password`：发件人的电子邮件密码（用于SMTP认证）。
        - `recipients`：收件人电子邮件地址的列表。
        - `subject`：电子邮件的主题。
        - `body`：电子邮件的正文内容。
    - **功能**：
        - 使用Gmail SMTP服务器向多个收件人发送电子邮件。

3. **连接到Gmail SMTP服务器并登录**
    ```python
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    ```
    - 使用 `smtplib.SMTP('smtp.gmail.com', 587)` 连接到Gmail的SMTP服务器，端口为587。
    - `server.starttls()` 启用TLS加密，确保连接的安全性。
    - `server.login(sender_email, sender_password)` 使用发件人的电子邮件地址和密码登录服务器。

4. **遍历每个收件人并创建邮件内容**
    ```python
    for recipient_email in recipients:
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        server.send_message(message)
    ```
    - 使用 `for` 循环遍历收件人列表 `recipients`，对每个收件人创建和发送电子邮件。
    - 使用 `MIMEMultipart()` 创建电子邮件对象 `message`。
    - 设置发件人地址、收件人地址和邮件主题。
    - `message.attach(MIMEText(body, 'plain'))` 将电子邮件正文内容以纯文本形式附加到邮件对象中。
    - `server.send_message(message)` 发送邮件给当前收件人。

5. **断开与SMTP服务器的连接**
    ```python
    server.quit()
    ```
    - 所有邮件发送完成后，使用 `server.quit()` 断开与SMTP服务器的连接。

6. **使用示例**
    ```python
    if __name__ == "__main__":
        sender_email = 'your_email@gmail.com'
        sender_password = 'your_password'
        recipients = ['recipient1@example.com', 'recipient2@example.com']
        subject = 'Hello'
        body = 'This is a test email.'
        send_personalized_email(sender_email, sender_password, recipients, subject, body)
    ```
    - 这部分代码确保当脚本作为主程序运行时执行。
    - 定义发件人的电子邮件地址和密码、收件人列表、邮件主题和正文。
    - 调用 `send_personalized_email` 函数发送邮件。

### **使用示例**

假设您有一个Gmail账户 `your_email@gmail.com`，并希望发送一封主题为“Hello”的邮件到多个收件人 `recipient1@example.com` 和 `recipient2@example.com`。运行脚本后，所有收件人将收到来自您邮箱的带有指定正文内容的邮件。

### **注意事项**

1. **隐私和安全性**
    - **密码保护**：脚本中使用发件人的密码进行登录。这是一个安全隐患，不建议直接在代码中写入明文密码。可以使用环境变量、加密或者应用密码（App Passwords）来增强安全性。
    - **Gmail安全设置**：默认情况下，Gmail可能会阻止通过应用程序的登录。因此，您可能需要启用Gmail账户的“允许不太安全的应用程序”选项，或者创建应用专用密码来进行认证。

2. **网络连接和端口**
    - 确保您的网络没有阻止端口587的连接，这是Gmail SMTP服务器的默认端口。

3. **TLS加密**
    - `server.starttls()` 用于启用TLS加密，确保与SMTP服务器之间的通信是安全的。强烈建议在发送敏感信息时使用TLS加密。

4. **邮件数量限制**
    - Gmail对于通过SMTP发送的邮件数量是有限制的。超过限额可能会导致您的账户被暂时锁定。建议不要一次性向过多的收件人发送邮件。

5. **异常处理**
    - 代码中没有异常处理，如果在连接、登录或发送邮件的过程中出现问题，脚本会直接报错。可以使用 `try-except` 结构添加异常处理，以提高代码的鲁棒性。

    示例代码添加异常处理：
    ```python
    def send_personalized_email(sender_email, sender_password, recipients, subject, body):
        try:
            # 连接到Gmail SMTP服务器并登录
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)

            # 遍历每个收件人并发送邮件
            for recipient_email in recipients:
                message = MIMEMultipart()
                message['From'] = sender_email
                message['To'] = recipient_email
                message['Subject'] = subject
                message.attach(MIMEText(body, 'plain'))
                server.send_message(message)
            print("所有邮件发送成功。")

        except smtplib.SMTPException as e:
            print(f"邮件发送失败: {e}")
        finally:
            # 确保无论如何都会断开与SMTP服务器的连接
            server.quit()
    ```

### **扩展功能建议**

1. **支持附件发送**
    - 可以扩展脚本支持发送带附件的邮件。例如，添加附件文件路径作为参数，并将附件附加到 `MIMEMultipart` 对象中。

    示例代码添加附件：
    ```python
    from email.mime.base import MIMEBase
    from email import encoders

    def send_personalized_email_with_attachment(sender_email, sender_password, recipients, subject, body, attachment_path):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

        for recipient_email in recipients:
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = recipient_email
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            # 添加附件
            attachment = MIMEBase('application', 'octet-stream')
            with open(attachment_path, 'rb') as attachment_file:
                attachment.set_payload(attachment_file.read())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename={attachment_path}')
            message.attach(attachment)

            # 发送邮件
            server.send_message(message)

        server.quit()
    ```

2. **个性化邮件内容**
    - 可以根据收件人的不同，动态调整邮件正文内容，增强个性化。例如，可以在邮件中包含收件人的姓名。

    示例代码个性化正文内容：
    ```python
    def send_personalized_email(sender_email, sender_password, recipients_info, subject, body_template):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

        for recipient_email, recipient_name in recipients_info.items():
            # 替换正文模板中的占位符，个性化邮件内容
            body = body_template.replace("{name}", recipient_name)

            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = recipient_email
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            server.send_message(message)

        server.quit()

    # 使用示例
    recipients_info = {
        'recipient1@example.com': 'Alice',
        'recipient2@example.com': 'Bob'
    }
    body_template = 'Hello {name},\nThis is a test email.'
    send_personalized_email(sender_email, sender_password, recipients_info, subject, body_template)
    ```

3. **使用环境变量保护敏感信息**
    - 使用 `os.environ` 从环境变量中获取发件人邮箱和密码，而不是在代码中明文保存这些敏感信息。

    示例代码：
    ```python
    import os

    sender_email = os.environ.get('SENDER_EMAIL')
    sender_password = os.environ.get('SENDER_PASSWORD')
    ```

4. **批量发送时的延时**
    - 如果需要批量发送较多邮件，可以添加适当的延时，避免触发Gmail的发送限制。

    示例代码添加延时：
    ```python
    import time

    for recipient_email in recipients:
        # ...（发送邮件代码）
        time.sleep(2)  # 每次发送后等待2秒
    ```

### **总结**

这个脚本是一个实用的工具，用于通过Gmail SMTP服务器向多个收件人发送电子邮件。它利用Python的 `smtplib` 和 `email` 库，提供了一个简单而有效的解决方案。为增强其灵活性和安全性，建议使用环境变量保存敏感信息，并添加适当的异常处理和日志记录。此外，您可以根据需要扩展功能，如支持附件、个性化邮件内容、增加发送延时等，以满足更多的使用场景。

在使用脚本时，请务必遵守邮件服务提供商的使用政策，避免滥用资源以及触发账户限制。此外，通过适当的安全措施来保护账户信息，确保数据安全。
