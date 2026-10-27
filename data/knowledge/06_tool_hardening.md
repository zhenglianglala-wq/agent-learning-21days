# 工具调用加固

主题：Agent 开发学习资料

工具执行器应统一处理 JSON 参数解析、未知工具、参数错误、超时、重试和日志。json.loads 把 JSON 字符串转换为 Python 字典。无论成功还是失败，返回结构应尽量统一，例如包含 ok、result、error_type、attempts 和 duration_ms。
