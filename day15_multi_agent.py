import json
import time
from datetime import datetime
from pathlib import Path

from day12_rag import MODEL, client, get_collection, record_usage, search_knowledge
from day13_agentic_rag import REWRITE_DISTANCE, run_agent

MAX_REVISIONS = 2
RESULTS_DIR = Path("logs")

def create_state(question: str) -> dict:
    return {
        "question": question,
        "research_results": [],
        "draft": "",
        "review": {},
        "revision_count": 0,
        "usage_records": [],
        "trace": [],
    }


def researcher(state: dict) -> None:
    state["trace"].append("researcher")
    collection = get_collection()
    results = search_knowledge(collection, state["question"], usage_records=state["usage_records"])
    state["research_results"] = [item for item in results if item["distance"] <= REWRITE_DISTANCE]

def writer(state: dict) -> None:
    state["trace"].append("writer")
    if not state["research_results"]:
        state["draft"] = "资料中未找到依据。"
        return
    context = "\n\n".join(
        f"[{item['filename']}#chunk-{item['chunk_id']}]\n{item['text']}"
        for item in state["research_results"])
    prompt = f"""
        你是报告撰写者。请根据参考资料回答用户问题。

        要求：
        1. 只能使用参考资料，不要编造。
        2. 按步骤组织回答。
        3. 每个关键结论附近标注来源，格式为 [文件名#chunk-编号]。
        4. 如果资料不足以回答某部分，请明确说明。

        用户问题：
        {state["question"]}

        参考资料：
        {context}
        审阅意见：
        {state["review"].get("feedback", "无")}
        """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    record_usage(state["usage_records"], "write", MODEL, response.usage)
    state["draft"] = response.choices[0].message.content or "未收到模型回答。"


def reviewer(state: dict) -> None:
    state["trace"].append("reviewer")
    context = "\n\n".join(
        f"[{item['filename']}#chunk-{item['chunk_id']}]\n{item['text']}"
        for item in state["research_results"])
    prompt = f"""
        你是报告审阅者。请根据用户问题和参考资料审阅报告。

        检查：
        1. 报告是否回答了用户问题。
        2. 报告中的结论是否有参考资料支持。
        3. 引用的来源是否与参考资料一致。

        资料不足时，报告应明确说明，不要要求作者编造内容。

        只输出 JSON，不要输出 Markdown：
        {{"approved": true, "feedback": "审阅意见"}}

        用户问题：
        {state["question"]}

        参考资料：
        {context}

        报告草稿：
        {state["draft"]}
        """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    record_usage(state["usage_records"], "review", MODEL, response.usage)
    text = response.choices[0].message.content or ""
    try:
        review = json.loads(text)
        if not isinstance(review, dict):
            raise ValueError("审阅结果必须是字典")

        if not isinstance(review.get("approved"), bool):
            raise ValueError("approved 必须是布尔值")

        if not isinstance(review.get("feedback"), str):
            raise ValueError("feedback 必须是字符串")

        state["review"] = review

    except (json.JSONDecodeError, ValueError):
        state["review"] = {
            "approved": False,
            "feedback": "审阅结果格式无效，需要重新检查。",
        }

def supervisor(state: dict) -> str:
    state["trace"].append("supervisor")

    if state["review"]["approved"]:
        return "finish"

    if state["revision_count"] >= MAX_REVISIONS:
        return "stop"

    return "revise"


def total_tokens(usage_records: list[dict]) -> int | None:
    if not usage_records:
        return None

    if any(item.get("total_tokens") is None for item in usage_records):
        return None

    return sum(item["total_tokens"] for item in usage_records)


def compare_with_single(state: dict, multi_duration: float) -> dict:
    start = time.perf_counter()
    single = run_agent(state["question"])
    single_duration = time.perf_counter() - start

    return {
        "question": state["question"],
        "single": {
            "answer": single["answer"],
            "duration_seconds": round(single_duration, 2),
            "total_tokens": total_tokens(single["usage_records"]),
        },
        "multi": {
            "answer": state["draft"],
            "review": state["review"],
            "revision_count": state["revision_count"],
            "trace": state["trace"],
            "duration_seconds": round(multi_duration, 2),
            "total_tokens": total_tokens(state["usage_records"]),
        },
    }
def save_comparison(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = RESULTS_DIR / f"day15_comparison_{timestamp}.json"

    file_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return file_path


def main():
    question = input("请输入报告主题：").strip()

    if not question:
        print("报告主题不能为空。")
        return

    state = create_state(question)
    multi_start = time.perf_counter()

    researcher(state)

    print("\n检索到的资料：")
    for item in state["research_results"]:
        print(f"\n来源：{item['filename']}#chunk-{item['chunk_id']}")
        print(f"距离：{item['distance']:.4f}")
        print(item["text"])

    print("\n执行链路：", state["trace"])
    print("模型用量：", state["usage_records"])

    writer(state)
    if state["research_results"]:
        reviewer(state)

        while True:
            action = supervisor(state)

            if action == "finish":
                print("\n审阅通过。")
                break

            if action == "stop":
                print("\n已达到修改上限，报告仍未通过审阅。")
                break

            state["revision_count"] += 1
            writer(state)
            reviewer(state)
    print("\n最终报告：")
    print(state["draft"])

    print("\n最终审阅意见：")
    print(state["review"])

    print("\n修改次数：", state["revision_count"])
    print("执行链路：", " -> ".join(state["trace"]))
    multi_duration = time.perf_counter() - multi_start
    result = compare_with_single(state, multi_duration)

    print("\n单 Agent 回答：")
    print(result["single"]["answer"])

    print("\n单 Agent 耗时：", result["single"]["duration_seconds"], "秒")
    print("单 Agent token：", result["single"]["total_tokens"])

    print("\n多角色耗时：", result["multi"]["duration_seconds"], "秒")
    print("多角色 token：", result["multi"]["total_tokens"])

    file_path = save_comparison(result)
    print("\n对比结果已保存到：", file_path)


if __name__ == "__main__":
    main()