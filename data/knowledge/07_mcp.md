# MCP Server 与 Client

主题：Agent 开发学习资料

MCP 是模型上下文协议，用于让客户端以统一方式发现和调用外部工具。Server 发布工具定义和实现，Client 通过 list_tools 发现工具，并用 call_tool 调用。MCP 解决工具接入标准问题，Function Calling 解决模型表达工具请求的问题。
