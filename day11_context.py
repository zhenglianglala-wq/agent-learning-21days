
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

DATABASE_PATH = Path("data/memory.sqlite")
THREAD_ID = "user-a-session-1"
USER_ID = "user-a"
RECENT_MESSAGE_LIMIT = 5
load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen3.8-flash"



def create_summary_table(connection: sqlite3.Connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            thread_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            covered_until_id INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    connection.commit()

def get_message_groups(connection: sqlite3.Connection, thread_id: str, recent_limit: int ) ->tuple[list[dict], list[dict]]:
    if recent_limit <= 0:
        raise ValueError("recent_limit must be a positive integer.")
    rows = connection.execute("""
        SELECT id, role, content FROM session_messages
        WHERE thread_id = ?
        ORDER BY id ASC
    """, (thread_id,)).fetchall()   
    messages = [{"id":message_id, "role": role, "content": content, } for message_id, role, content in rows]
    old_messages = messages[:-recent_limit]
    recent_messages = messages[-recent_limit:]
    return old_messages, recent_messages

def get_existing_summary(connection: sqlite3.Connection, thread_id: str) -> dict | None:
    row = connection.execute("""
        SELECT summary, covered_until_id, updated_at FROM session_summaries
        WHERE thread_id = ?
    """, (thread_id,)).fetchone()
    if row is None:
        return None
    summary, covered_until_id, updated_at = row
    return {
        "summary": summary,
        "covered_until_id": covered_until_id,
        "updated_at": updated_at
    }

async def summarize_messages(messages: list[dict], existing_summary: dict | None = None ) -> str:
    conversation_blocks = [
        f"{msg['role']}: {msg['content']}" for msg in messages
    ]
    new_messages_text = "\n".join(conversation_blocks)
    source_text = ""
    if existing_summary:
        source_text += (
            "已有摘要（代表更早的历史，请保留其中关键信息）：\n"
            f"{existing_summary}\n\n"
        )
    source_text += (
        "新增的旧消息：\n"
        f"{new_messages_text}"
    )

    summary_prompt = (
        "请将下面的已有摘要和新增旧消息合并成一份新的精简摘要。\n"
        "保留用户偏好、重要事实、技术需求、已完成事项和关键约定。\n"
        "禁止编造，不要输出标题或 Markdown，只输出摘要正文。\n\n"
        f"{source_text}"
    )
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": summary_prompt},
        ],
    )
    return response.choices[0].message.content or "No summary generated."

   

def upsert_summary(connection: sqlite3.Connection, thread_id: str, summary: str, covered_until_id: int):
    updated_at = datetime.now(timezone.utc).isoformat()
    connection.execute("""
        INSERT INTO session_summaries (thread_id, summary, covered_until_id, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            summary=excluded.summary,
            covered_until_id=excluded.covered_until_id,
            updated_at=excluded.updated_at
    """, (thread_id, summary, covered_until_id, updated_at))
    connection.commit()



def retrieve_related_messages(connection: sqlite3.Connection, thread_id: str, query_text: str, recent_limit: int) -> list[dict[str, str]]:
    if recent_limit <= 0:
        raise ValueError("recent_limit must be a positive integer.")
    rows = connection.execute("""
        SELECT  id, role, content FROM session_messages
        WHERE thread_id = ? 
        ORDER BY id ASC
    """, (thread_id,)).fetchall()
    all_messages = [{"id": message_id, "role": role, "content": content} for message_id, role, content in rows]
    query_tokens = query_text.lower().split()
    matched = []
    for msg in all_messages:
        content_lower = msg["content"].lower()
        for token in query_tokens:
            if token and token in content_lower:
                matched.append(msg)
                break
    return matched[-recent_limit:]


def build_final_context(
    connection: sqlite3.Connection,
    user_id: str,
    thread_id: str,
    final_summary_text: str | None,
    related_messages: list[dict],
    recent_messages: list[dict],
    old_messages: list[dict]
) -> tuple[list[dict[str, str]], int, int]:
    """
    组装送入大模型的完整消息上下文，同时统计字符长度
    :param connection: sqlite连接
    :param user_id: 用户ID，用于读取用户偏好构建system prompt
    :param thread_id: 会话ID
    :param final_summary_text: 旧消息的摘要文本，None代表无摘要
    :param related_messages: 关键词检索得到的相关历史消息
    :param recent_messages: 最近N条原始消息，原样放入上下文
    :return: (model_messages, original_total_chars, compressed_total_chars)
             model_messages：给模型的消息列表
             original_total_chars：原始全部旧消息总字符数
             compressed_total_chars：摘要的字符数
    """
    # 读取用户偏好，构建系统提示词（沿用原有build_system_prompt逻辑）
    def list_preferences(conn: sqlite3.Connection, uid: str) -> dict[str, str]:
        rows = conn.execute("""
            SELECT preference_key, preference_value FROM user_preferences
            WHERE user_id = ?
            ORDER BY preference_key
        """, (uid,)).fetchall()
        return {k: v for k, v in rows}

    def build_system_prompt(preference: dict[str, str]) -> str:
        prompt = "你是一个个人知识库助手。"
        language = preference.get("language")
        if language:
            prompt += f" 用户的语言偏好是 {language}。"
        response_style = preference.get("response_style")
        if response_style:
            prompt += f" 用户的回答风格偏好是 {response_style}。"
        return prompt

    pref = list_preferences(connection, user_id)
    system_prompt = build_system_prompt(pref)

    model_messages: list[dict[str, str]] = []
    model_messages.append({"role": "system", "content": system_prompt})
    for message in related_messages:
        model_messages.append({
        "role": message["role"],
        "content": message["content"],
    })

    for message in recent_messages:
        model_messages.append({
        "role": message["role"],
        "content": message["content"],
    })

    # 统计原始旧消息总字符
    original_total_chars = 0
    # 统计摘要字符
    compressed_total_chars = 0

    # 添加旧消息摘要（如果存在）
    if final_summary_text is not None and final_summary_text.strip():
        model_messages.append({
            "role": "system",
            "content": f"【更早对话历史摘要】：{final_summary_text}"
        })
        compressed_total_chars = len(final_summary_text)

    original_old_chars = sum(
    len(message["content"])
    for message in old_messages)

    final_context_chars = sum(
        len(message["content"])
        for message in model_messages
    )

    return model_messages, original_old_chars, final_context_chars


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


async def ask_model(messages: list[dict[str, str]]) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    answer = response.choices[0].message.content or "No answer received."
    return answer


async def main():
    with sqlite3.connect(DATABASE_PATH) as connection:
        create_summary_table(connection)
        old_messages, recent_messages = get_message_groups(connection, THREAD_ID, RECENT_MESSAGE_LIMIT)
        print(f"旧消息数量：{len(old_messages)}")
        print(f"最近消息数量：{len(recent_messages)}")

        summary_record = get_existing_summary(connection, THREAD_ID)
        last_covered_id = (summary_record["covered_until_id"] if summary_record else 0)
        new_need_summarize = [message for message in old_messages if message["id"] > last_covered_id]
        if new_need_summarize:
            print(f"需要合并摘要 {len(new_need_summarize)} 条旧消息")
            existing_summary = (summary_record["summary"] if summary_record else None)
            final_summary_text = await summarize_messages(new_need_summarize, existing_summary=existing_summary)
            covered_until_id = new_need_summarize[-1]["id"]
            upsert_summary(connection, THREAD_ID, final_summary_text, covered_until_id)
        else:
            final_summary_text = (summary_record["summary"] if summary_record else None)
            print("没有新增旧消息，复用已有摘要")
        # 接收用户输入
        user_question = input("请输入用户消息：").strip()
        if not user_question:
            print("用户消息不能为空。")
            return

        # 写入用户提问到消息表
        add_message(connection, THREAD_ID, "user", user_question)
        _,recent_messages = get_message_groups(connection, THREAD_ID, RECENT_MESSAGE_LIMIT)
        retrieved_messages = retrieve_related_messages(connection, THREAD_ID, user_question, RECENT_MESSAGE_LIMIT)
        recnet_ids = [message["id"] for message in recent_messages]
        related_messages = [message for message in retrieved_messages if message["id"] not in recnet_ids]
        recent_slim = [{"role": msg["role"], "content": msg["content"]} for msg in recent_messages]
        related_slim = [{"role": msg["role"], "content": msg["content"]} for msg in related_messages]
        print(f"检索到相关旧消息条数：{len(related_messages)}")

        # 组装最终模型消息、统计字符
        model_messages, original_chars, compressed_chars = build_final_context(
            connection=connection,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            final_summary_text=final_summary_text,
            related_messages=related_slim,
            recent_messages=recent_slim,
            old_messages=old_messages,
        )

        print(f"\n旧消息原文字符数：{original_chars}")
        print(f"最终发送上下文字符数：{compressed_chars}")
        for msg in model_messages:
            print(msg)

        # 调用模型获取回答
        answer = await ask_model(model_messages)
        # 保存assistant回复
        add_message(connection, THREAD_ID, "assistant", answer)
        print(f"\n模型回复：{answer}")

if __name__ == "__main__":
    asyncio.run(main())