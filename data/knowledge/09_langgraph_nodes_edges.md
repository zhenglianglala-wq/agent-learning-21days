# LangGraph Node 与 Edge

主题：Agent 开发学习资料

Node 是处理 State 的函数，例如 call_model 或 run_tool。Edge 决定节点执行顺序。Conditional Edge 根据当前 State 选择下一节点，例如模型要求工具时去 run_tool，模型给出 final 时去 END。图需要 compile 后才能 invoke。
