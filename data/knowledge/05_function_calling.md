# Function Calling

主题：Agent 开发学习资料

Function Calling 让模型以 tool_calls 的结构提出工具调用请求。工具 schema 告诉模型工具名、用途和参数格式；本地 TOOL_FUNCTIONS 字典把工具名映射到实际 Python 函数。程序需要解析参数、执行函数，并将 role=tool 的结果消息传回模型。
