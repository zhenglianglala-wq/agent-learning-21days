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
    
    result = operations[operation](x, y)
    return {"operation": operation, "x": x, "y": y, "result": result}
async def agent_loop(messages: list[dict], max_rounds: int = 10) ->dict:
    for round in range(max_rounds):    
        response = await client.chat.completions.create(
            model="qwen3.8-flash",
            messages=messages,
        )
        model_text = response.choices[0].message.content
        action = json.loads(model_text)
        if action.get("type") == "calculation":
            x = action.get("x")
            y = action.get("y")
            operation = action.get("operation")
            result = run_calculation(x, y, operation)
            messages.append({"role": "assistant", "content": model_text})
            messages.append({"role": "user", "content": f"工具执行结果：{json.dumps(result, ensure_ascii=False)}。请继续完成任务。", "name": "tool_result"})
            continue
        elif action.get("type") == "final":
            return action
    else:
        return {"type": "final", "answer": "Reached maximum rounds without a final answer."}

def main(max_rounds: int = 10):
    print("Welcome to the Mathematical Agent!")
    contents = input("Enter your mathematical operation (e.g., 'add 5 3') or 'exit' to quit: ")
    if contents.lower() == "exit":
        print("Exiting the agent loop.")
        return
    messages = [
        {
            "role": "system",
            "content":  """你是数学 Agent，只能输出一个 JSON 对象，不能输出 Markdown 或其他文字。需要调用计算工具时，输出：{\"type\": \"calculation\", \"operation\": \"add\", \"x\": 12, \"y\": 8}拿到全部计算结果后，输出：{"type": "final", "answer": "最终结果是 40。"}可用 operation：add、subtract、multiply、divide。每次只能输出一个动作。"""},
        {"role": "user", "content": contents}
    ]
    result = asyncio.run(agent_loop(messages, max_rounds))
    print(f"Final result: {result}")
        
if __name__ == "__main__":
    main()