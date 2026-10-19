import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
import json
import asyncio


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
PRACTICE_DIR = Path(__file__).resolve().parent /"data" / "practice"
PRACTICE_DIR.mkdir(parents=True, exist_ok=True)

def get_safe_path(filename: str) -> Path:
    root = PRACTICE_DIR.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Invalid filename: {filename}. It must be a relative path within the practice directory.") from e
    return candidate

def list_files() -> dict:
    """
    List all files in the practice directory.
    """
    files = [path.name for path in PRACTICE_DIR.iterdir() if path.is_file()]
    return {"files": files, "count": len(files)}

def read_file(filename: str) -> dict:
    path = get_safe_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    elif not path.is_file():
        raise ValueError(f"Path is not a file: {filename}")
    else:
        return {"filename": filename, "content": path.read_text(encoding="utf-8")}

def search_files(keyword: str) -> dict:
    """
    Search for files containing the given keyword in the practice directory.
    """
    if not keyword.strip():
        raise ValueError("Keyword must not be empty.")
    needle = keyword.lower()
    matching_files = []
    for path in PRACTICE_DIR.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").lower()
        if needle in path.name.lower() or needle in content.lower():
            matching_files.append(
                path.relative_to(PRACTICE_DIR).as_posix())
    return {"keyword": keyword, "matching_files": matching_files, "count": len(matching_files)}


"""
    path = get_safe_path(keyword)
    if not keyword:
        raise ValueError("Keyword must not be empty.")
    PRACTICE_DIR.rglob("*")  # Ensure the directory is scanned
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    for keyword in path.name.lower():
        matching_files = []
        return {"keyword": keyword, "matching_files": matching_files, "count": len(matching_files)}
    else:
        file_contents = path.read_text(encoding="utf-8")
        for keyword in file_contents.lower():
            matching_files = []
            return {"keyword": keyword, "matching_files": matching_files, "count": len(matching_files)}
    matching_files = []
    for path in PRACTICE_DIR.iterdir():
        if path.is_file() and keyword in path.name:
            matching_files.append(path.name)
    return {"keyword": keyword, "matching_files": matching_files, "count": len(matching_files)}

"""


def save_note(filename: str, content: str) -> dict:
    """
    Save content to a file in the practice directory.
    """
    path = get_safe_path(filename)
    if not content.strip():
        raise ValueError("Content must not be None.")  
    if path.exists():
        old_content = path.read_text(encoding="utf-8")
        if old_content == content:
            return {"filename": path.name, "status": "unchanged", "characters_written": len(content)}
        status = "updated"
    else:
        status = "created"
    path.write_text(content, encoding="utf-8")
    return {"filename": path.name, "status": status, "characters_written": len(content)}



def calculate(operation: str, x: float, y: float) -> dict:
    """
    A simple function that can perform basic mathematical operations.
    """
    if operation not in ["add", "subtract", "multiply", "divide"]:
        raise ValueError(f"Invalid operation: {operation}. Supported operations are: add, subtract, multiply, divide.")
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    
    result = operations[operation](x, y)
    return {"operation": operation, "x": x, "y": y, "result": result}   
TOOLS =[
{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行加减乘除计算。",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["operation", "x", "y"],
            "additionalProperties": False,
        },
    },
} ,
{
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "保存笔记到文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "列出练习目录中的所有文件。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取指定文件的内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "搜索包含指定关键字的文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
}
]
TOOL_FUNCTIONS = {   
    "calculate": calculate,
    "save_note": save_note,
    "list_files": list_files,
    "read_file": read_file,
    "search_files": search_files,
}

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
            arguments = json.loads(tool_call.function.arguments)
            tool_function = TOOL_FUNCTIONS.get(tool_name)
            if tool_function is None:
                tool_result = {"error": f"Tool '{tool_name}' is not implemented."}
            else:
                try:
                    tool_result = tool_function(**arguments)
                except Exception as e:
                    tool_result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result, ensure_ascii=False)})
    return "Reached maximum rounds without a final answer."



def main():
    print("Welcome to the Function Calling Agent!")
    user_request = input("Enter your request (or 'exit' to quit): ")
    if user_request.lower() == "exit":
        print("Exiting the agent loop.")
        return
    messages = [
        {
            "role": "system","content":"你可以使用提供的工具完成任务。"},
        {"role": "user", "content": user_request}
    ]
    response = asyncio.run(run_agent(messages,10))
    print(f"Model response: {response}")

if __name__ == "__main__":
    main()
