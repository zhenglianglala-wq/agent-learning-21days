from typing import TypedDict
from langgraph.graph import END, START, StateGraph
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
import json

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")    
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
MAX_ROUNDS = 10


class AgentState(TypedDict):
    user_request: str
    messages: list[dict]
    round_count: int

async def call_model(state: AgentState) -> dict:
    print("Node: call_model")

    response = await client.chat.completions.create(
        model="qwen3.8-flash",
        messages=state["messages"],
    )
    model_text = response.choices[0].message.content
    print("Model output:", model_text)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": model_text}], "round_count": state["round_count"] + 1
    }


def route_after_model(state: AgentState) -> str:
    print("Node: route_after_model")
    if state["round_count"] >= MAX_ROUNDS:
        return END
    model_text = state["messages"][-1]["content"]
    try:
        action = json.loads(model_text)
    except json.JSONDecodeError:
        return END
    if action.get("type") == "calculation":
        return "run_tool"

    return END

def run_calculation(x: float, y: float, operation: str) -> dict:
    """
    A simple agent that can perform basic mathematical operations.
    """
    # Define the operations the agent can perform
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
    if operation == "divide" and y == 0:
        raise ValueError("Division by zero is not allowed.")
    return {"operation": operation, "x": x, "y": y, "result": operations[operation](x, y)}

async def run_tool(state: AgentState) -> dict:
    print("Node: run_tool")

    action = json.loads(state["messages"][-1]["content"])
    if action.get("type") != "calculation":
        raise ValueError("Expected a calculation action.")
    result = run_calculation(
        x = action["x"],
        y = action["y"],
        operation = action["operation"]
    )
    tool_message = {
        "role": "user",
        "name": "tool_result",
        "content": f"工具执行结果：{json.dumps(result, ensure_ascii=False)}。请继续完成任务。"
    }
    return {
        "messages": state["messages"] + [tool_message]}


workflow = StateGraph(AgentState)
workflow.add_node("call_model", call_model)
workflow.add_node("run_tool", run_tool)
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges(
    "call_model",
    route_after_model,
    {
        "run_tool": "run_tool",
        END: END
    }
)

workflow.add_edge("run_tool", "call_model")
graph = workflow.compile()

async def main() -> None:
    print("Welcome to the Mathematical Agent!")
    user_request = input("Enter your mathematical operation (e.g., 'add 5 3') or 'exit' to quit: ")
    if user_request.lower() == "exit":
        print("Exiting the agent loop.")
        return
    initial_state: AgentState = {
        "user_request": user_request,
        "messages": [
            {
                "role": "system",
                "content": """
                    你是数学 Agent，只能输出一个 JSON 对象，不能输出其他文字。

                    你绝对不能自行计算。每一个加减乘除步骤都必须请求计算工具。

                    规则：
                    1. 需要计算时，只输出：
                    {"type":"calculation","operation":"add","x":12,"y":8}

                    2. 收到“工具执行结果”后：
                    - 若任务还有计算步骤，继续输出下一次 calculation；
                    - 所有计算步骤完成后，才输出：
                    {"type":"final","answer":"最终结果是 40。"}

                    3. 每次只能输出一个动作。
                    可用 operation：add、subtract、multiply、divide。
                    """,
            },
            {"role": "user", "content": user_request}
        ],
        "round_count": 0,
    }
    final_state = await graph.ainvoke(initial_state)
    final_text = final_state["messages"][-1]["content"]
    print("Final output from the agent:", final_text)

if __name__ == "__main__":
    asyncio.run(main())