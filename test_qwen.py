import os
from openai import OpenAI

try:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3.8-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "where is the capital of canada?"},
        ],
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
