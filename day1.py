import os
from pathlib import Path
from typing import Optional
import dashscope
from dashscope import Generation
from dotenv import load_dotenv


def get_api_key() -> Optional[str]:
    """Read the key from the shell environment or the local .env file."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key.strip()

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("DASHSCOPE_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'") or None
        # Also accept the current .env format, where the file contains only the key.
        if "=" not in line:
            return line
    return None

def main():
    # Always load the .env next to this script, regardless of the terminal directory.
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    api_key = get_api_key()
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable is not set.")
        return
    dashscope.base_http_api_url = "https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/api/v1"
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "where is the capital of canada?"},
    ]
    response = Generation.call(
        model="qwen-plus",
        messages=message,
        api_key=api_key,
        result_format="message",
    )
    if response.status_code == 200:
        answer = response.output.choices[0].message.content
        print(f"Answer: {answer}")
    else:
        print(f"Request failed: status={response.status_code}")
        print(f"Error code: {response.code}")
        print(f"Error message: {response.message}")
if __name__ == "__main__":
        main()
