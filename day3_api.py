from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import AsyncOpenAI  


# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

app = FastAPI(title="Chat API", description="A simple chat API using FastAPI.", version="1.0.0")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000, description="The message to send to the model.")

class ChatResponse(BaseModel):
    answer: str = Field(description="The answer from the model.")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    response = await client.chat.completions.create(
        model="qwen3.8-flash",
        messages=[
            {"role": "user", "content": request.message},
        ],
    )

    answer = response.choices[0].message.content or "No answer received."
    return ChatResponse(answer=answer)
