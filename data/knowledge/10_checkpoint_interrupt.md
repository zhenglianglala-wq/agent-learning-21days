# Checkpoint 与 Interrupt

主题：Agent 开发学习资料

LangGraph checkpointer 会按 thread_id 保存图的执行状态。interrupt 会暂停工作流并向外返回等待信息。程序重启后，使用相同 thread_id 和 Command(resume=...) 可以从暂停点恢复。写文件等有副作用的操作应在确认后执行。
