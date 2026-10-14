import os
import openai
from openai import OpenAI
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")


client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

async def ask_qwen(question: str, timeout_seconds: float) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
            model="qwen3.8-flash",
                messages=messages,
            ), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return {"question": question,"status": "timeout","answer": None,}
    except Exception as e:
        return {"question": question,"status": "error","answer": None,"error": str(e),}
    return {"question": question,"status": "success","answer": response.choices[0].message.content,}
async def main():
    requests = [
        ("用一句话解释 async/await。", 5),
        ("用一句话解释 API。", 5),
        ("用一句话解释 JSON。", 0.001),
    ]
    tasks = [
        asyncio.create_task(ask_qwen(question, timeout_seconds)) for question, timeout_seconds in requests
    ]

    results = await asyncio.gather(*tasks)
    for result in results:
        if result["status"] == "timeout":
            print(f"Question: {result['question']}\nStatus: Timeout\n")
        elif result["status"] == "error":
            print(f"Question: {result['question']}\nStatus: Error\nError: {result['error']}\n")
        else:
            print(f"Question: {result['question']}\nAnswer: {result['answer']}\n")

if __name__ == "__main__":
    asyncio.run(main())