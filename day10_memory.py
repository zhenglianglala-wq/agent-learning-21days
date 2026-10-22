from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen3.8-flash"

DATABASE_PATH = Path("data/memory.sqlite")
THREAD_ID = "day10-demo-thread"
USER_ID = "user-a"
THREAD_ID = "user-a-session-2"


def create_tables(connection: sqlite3.Connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT NOT NULL,
            preference_key TEXT NOT NULL,
            preference_value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, preference_key)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    connection.commit()

def set_preference(connection: sqlite3.Connection, user_id: str, preference_key: str, preference_value: str):
    updated_at = datetime.now(timezone.utc).isoformat()
    connection.execute("""
        INSERT INTO user_preferences (user_id, preference_key, preference_value, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, preference_key) DO UPDATE SET
            preference_value=excluded.preference_value,
            updated_at=excluded.updated_at
    """, (user_id, preference_key, preference_value, updated_at))
    
    connection.commit()

def get_preference(connection: sqlite3.Connection, user_id: str, preference_key: str) -> str | None:
    row = connection.execute("""
        SELECT preference_value FROM user_preferences
        WHERE user_id = ? AND preference_key = ?
    """, (user_id, preference_key)).fetchone()
    if row is None:
        return None
    return row[0]

def list_preferences(connection: sqlite3.Connection, user_id: str) -> dict[str, str]:
    rows = connection.execute("""
        SELECT preference_key, preference_value FROM user_preferences
        WHERE user_id = ?
        ORDER BY preference_key
    """, (user_id,)).fetchall()
    return {
        perference_key: preference_value
        for perference_key, preference_value in rows
    }

def build_system_prompt(preference: dict[str, str]) -> str:
    prompt = "你是一个个人知识库助手。"

    language = preference.get("language")
    if language:
        prompt += f" 用户的语言偏好是 {language}。"
    response_style = preference.get("response_style")
    if response_style:
        prompt += f" 用户的回答风格偏好是 {response_style}。"

    return prompt

def add_message(connection: sqlite3.Connection, thread_id: str, role: str, content: str):
    allowed_roles = {"user", "assistant", "system", "tool"}
    if role not in allowed_roles:
        raise ValueError(f"Invalid role: {role}. Allowed roles are: {allowed_roles}")
    created_at = datetime.now(timezone.utc).isoformat()
    connection.execute("""
        INSERT INTO session_messages (thread_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (thread_id, role, content, created_at))
    connection.commit()

def get_recent_messages(connection: sqlite3.Connection, thread_id: str, limit: int=5) -> list[dict[str, str]]:
    if limit <= 0:
        raise ValueError("Limit must be a positive integer.")
    rows = connection.execute("""
        SELECT role, content, created_at FROM session_messages
        WHERE thread_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (thread_id, limit)).fetchall()
    rows.reverse()  # To return messages in chronological order
    return [
        {"role": role, "content": content}
        for role, content, created_at in rows
    ]

def build_model_messages(connection: sqlite3.Connection, user_id: str, thread_id: str) -> list[dict[str, str]]:
    preference = list_preferences(connection, user_id)
    system_prompt = build_system_prompt(preference)
    history = get_recent_messages(connection, thread_id)
    return [
        {"role": "system", "content": system_prompt},
        *history,]

async def ask_model(messages: list[dict[str, str]]) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    answer = response.choices[0].message.content or "No answer received."
    return answer

def seed_demo_preferences(connection: sqlite3.Connection):
    if get_preference(connection, USER_ID, "language") is None:
        set_preference(connection, USER_ID, "language", "中文")
    if get_preference(connection, USER_ID, "response_style") is None:
        set_preference(connection, USER_ID, "response_style", "简洁")



async def main():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        create_tables(connection)
        seed_demo_preferences(connection)

        
        
        print("已初始化演示偏好。")
        print(f"当前用户：{USER_ID}")
        print(f"当前会话：{THREAD_ID}")
        user_question = input("请输入用户消息：").strip()
        if not user_question:
            print("用户消息不能为空。")
            return
        add_message(connection, THREAD_ID, "user", user_question)
        print(f"已保存 {THREAD_ID} 的一条用户消息")
        history = get_recent_messages(connection, THREAD_ID)
        print("最近会话历史：")
        for message in history:
            print(message)
        messages = build_model_messages(connection, "user-a", THREAD_ID)
        print("构建的模型消息：")
        for message in messages:
            print(message)

        answer = await ask_model(messages)
        add_message(connection, THREAD_ID, "assistant", answer)
        print(f"已保存 {THREAD_ID} 的一条模型回复")
        print(f"模型回复：{answer}")


if __name__ == "__main__":
    asyncio.run(main())