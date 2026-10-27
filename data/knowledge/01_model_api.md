# 模型 API 与环境变量

主题：Agent 开发学习资料

模型 API 的密钥应放在 .env 文件中，通过环境变量读取。代码不应把 API Key 直接写死，也不应提交 .env 到 Git。OpenAI 兼容接口通常使用 client.chat.completions.create 发送 messages，并从 response.choices[0].message.content 读取回答。
