# LangGraph State

主题：Agent 开发学习资料

LangGraph 用 State 保存工作流共享数据，例如 messages、round_count、tool_result。每个节点接收当前 State，并返回 State 的更新。State 让循环、条件分支和持久化流程都有明确的数据边界。
