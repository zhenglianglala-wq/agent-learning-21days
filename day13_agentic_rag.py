import json

from day12_rag import (MODEL, client, get_collection, search_knowledge, answer_with_rag)







MAX_SEARCH_ATTEMPTS = 2
REWRITE_DISTANCE = 0.55



def create_agent_state(question: str) -> dict:
    return {
        "question": question,
        "search_query": question,
        "search_attempts": 0,
        "results": [],
        "route": None,
        "answer": None,
        "trace": [],
        "planning_usage": None,
        "rewrite_usage": None,
        "answer_usage": None,
        "usage_records": [],
    }

def decide_initial_action(question: str) -> str:
    prompt = f"""你是知识库 Agent 的规划器。根据用户问题，只能选择一个 action：
    - direct_answer：问候、简单交流，不需要本地资料。
    - search：技术事实、学习内容、需要资料依据的问题。

    只输出 JSON，不要 Markdown：
    {{"action": "direct_answer 或 search", "reason": "简短原因"}}

    用户问题：{question}
    """
    response = client.chat.completions.create(
        model = MODEL,
        messages = [{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or "未收到模型回答。"
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        decision = {"action": "search", "reason": "模型输出不是有效 JSON，默认检索。"}

    if not isinstance(decision, dict) or decision.get("action") not in {"direct_answer", "search"}:
        decision = {"action": "search", "reason": "未知动作，默认检索。"}
    usage = response.usage
    decision["usage"] = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    } if usage else None
    return decision

def answer_directly(question: str) -> str:
    response = client.chat.completions.create(
        model = MODEL,
        messages = [
            {
                "role": "system",
                "content": "你是学习助手，请用中文简洁回答。",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    usage = response.usage
    usage_data = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    } if usage else None

    return response.choices[0].message.content or "未收到模型回答。", usage_data

def run_agent(question: str) -> dict:
    state = create_agent_state(question)
    collection = get_collection()
    decision = decide_initial_action(question)
    state["planning_usage"] = decision.get("usage")
    state["usage_records"].append({"stage": "planning", "model": MODEL, **(state["planning_usage"] or {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})})

    if decision["action"] == "direct_answer":
        state["route"] = "direct_answer"
        state["answer"], state["answer_usage"] = answer_directly(question)
        state["usage_records"].append({"stage": "direct_answer", "model": MODEL, **(state["answer_usage"] or {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})})
        state["trace"].append("direct_answer")
        return state

    while state["search_attempts"] < MAX_SEARCH_ATTEMPTS:

        state["route"] = "search"
        state["trace"].append(f"search:{state['search_query']}")
        state["results"] = search_knowledge(collection, state["search_query"], usage_records=state["usage_records"])
        state["search_attempts"] += 1
        if results_are_sufficient(state["results"]):
            state["route"] = "search_with_results"
            state["answer"] = answer_with_rag(state["question"], state["results"], usage_records=state["usage_records"])
            state["answer_usage"] = {key: state["usage_records"][-1][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
            return state
        if state["search_attempts"] >= MAX_SEARCH_ATTEMPTS:
            state["route"] = "no_evidence"
            state["answer"] = "资料中未找到依据。"
            return state
        state["route"] = "rewrite_query"
        # 改写查询前
        state["trace"].append("rewrite_query")
        state["search_query"], state["rewrite_usage"] = rewrite_query(state["question"], state["search_query"])
        state["usage_records"].append({"stage": "rewrite", "model": MODEL, **(state["rewrite_usage"] or {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})})
        state["trace"].append(f"new_query:{state['search_query']}")
    return state






def rewrite_query(
    original_question: str,
    previous_query: str,
) -> str:
    prompt = f"""
    你是知识库检索查询改写器。

    将用户问题改写成适合检索的关键词，遵守以下规则：
    1. 保留用户问题中的核心对象、主题和限制条件。
    2. 不要添加用户没有提到的主题，例如 Agent、工具或 API。
    3. 不要回答问题，也不要改变用户原本的意图。
    4. 如果无法改写，就原样返回用户问题。
    不要回答问题，只输出 JSON：

    {{"query": "改写后的查询词"}}

    用户原始问题：{original_question}
    上一次查询词：{previous_query}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content or ""

    try:
        data = json.loads(text)
        new_query = data.get("query", "").strip()
    except json.JSONDecodeError:
        new_query = ""

    usage = response.usage
    usage_data = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    } if usage else None

    return new_query or previous_query, usage_data


def results_are_sufficient(results: list[dict]) -> bool:
    if not results:
        return False
    best_distance = min(item["distance"] for item in results)
    return best_distance <= REWRITE_DISTANCE




def main():
    while True:
        question = input("请输入问题（或输入 'exit' 退出）：").strip()
        if question.lower() == "exit":
            break
        if not question:
            print("问题不能为空。")
            continue

        result = run_agent(question)
        print(f"\n路径：{result['route']}")
        print(f"最终查询词：{result['search_query']}")
        print(f"检索次数：{result['search_attempts']}")
        print(f"\n回答：{result['answer']}")
        print("执行链路：")
        print(" -> ".join(result["trace"]))

if __name__ == "__main__":
    main()
