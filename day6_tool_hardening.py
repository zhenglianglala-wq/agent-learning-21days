import asyncio
import json
from day5_function_calling import TOOL_FUNCTIONS, TOOLS
import time
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI


TOOL_TIMEOUT_SECONDS = 3
MAX_TOOL_ATTEMPTS = 3
LOG_FILE = Path(__file__).resolve().parent / "logs" / "tool_calls.jsonl"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # Ensure the logs directory exists
PRACTICE_DIR = Path(__file__).resolve().parent /"data" / "practice"
PRACTICE_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

async def execute_tool(tool_name: str, arguments_json:str) -> dict:

    started_at = time.perf_counter()
    
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return finish({"ok": False, "tool_name": tool_name, "error_type": "invalid_json", "message": "Arguments must be valid JSON."}, started_at)
    
    if not isinstance(arguments, dict):
        return finish({"ok": False, "tool_name": tool_name, "error_type": "invalid_arguments", "message": "Arguments must be a JSON object."}, started_at)
    
    tool_function = TOOL_FUNCTIONS.get(tool_name)
    

    if tool_function is None:
        return finish({"ok": False, "tool_name": tool_name, "error_type": "unknown_tool", "message": f"Tool '{tool_name}' is not implemented."}, started_at)

    for attempt in range(1, MAX_TOOL_ATTEMPTS + 1):
        try:
            result = await asyncio.wait_for(asyncio.to_thread(tool_function, **arguments), timeout=TOOL_TIMEOUT_SECONDS)
            return finish({"ok": True,"tool_name": tool_name, "data": result, "attempts": attempt}, started_at)
        except asyncio.TimeoutError:
            if attempt < MAX_TOOL_ATTEMPTS:
                continue
        except TypeError as e:
            return finish({"ok": False, "tool_name": tool_name, "error_type": "invalid_arguments", "message": str(e)}, started_at)
        except (ValueError, FileNotFoundError) as e:    
            return finish({"ok": False, "tool_name": tool_name, "error_type": "tool_error", "message": str(e),}, started_at)
    return finish({"ok": False, "tool_name": tool_name, "error_type": "timeout", "message": "工具调用超时", "attempts": attempt, "max_attempts": MAX_TOOL_ATTEMPTS}, started_at)



def slow_tool(seconds: float) -> str:
    """
    A tool that simulates a slow operation by sleeping for the specified number of seconds.
    """
    time.sleep(seconds)
    return f"Slept for {seconds} seconds."


TOOL_FUNCTIONS["slow_tool"] = slow_tool  # Register the slow tool function

def log_tool_call(record: dict) -> None:
    """
    Log the tool call record to a JSONL file.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool_name": record.get("tool_name"),
        "ok": record.get("ok"),
        "error_type": record.get("error_type"),
        "attempts": record.get("attempts"),
        "duration_ms": record.get("duration_ms"),
    }
    record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def finish(result: dict ,started_at: float) -> dict:
    """
    Finish the tool call by logging the result and returning it.
    """
    result["duration_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    log_tool_call(result)
    return result


async def ask_model(messages:list[dict]):
    response = await client.chat.completions.create(
        model="qwen3.8-flash",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    message = response.choices[0].message
    print(message.tool_calls)
    return message

async def run_agent(messages: list[dict], max_rounds: int = 10) -> str:
    for _ in range(max_rounds):
        message = await ask_model(messages)
        if not message.tool_calls:
            return message.content or "No answer received."
        messages.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_result = await execute_tool(tool_name, tool_call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result, ensure_ascii=False)})
    return "Reached maximum rounds without a final answer."


def main():
    print("Welcome to the Function Calling Agent!")
    while True:
        user_request = input("Enter your request (or 'exit' to quit): ")
        if user_request.lower() == "exit":
            print("Exiting the agent loop.")
            break
        messages = [
            {
                "role": "system","content":"你可以使用提供的工具完成任务。"},
            {"role": "user", "content": user_request}
        ]
        response = asyncio.run(run_agent(messages,10))
        print(f"Model response: {response}")

if __name__ == "__main__":
    main()