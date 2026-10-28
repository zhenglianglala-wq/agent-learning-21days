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
        return {"action": "search", "reason": "模型输出不是有效 JSON，默认检索。"}

    if decision.get("action") not in {"direct_answer", "search"}:
        return {"action": "search", "reason": "未知动作，默认检索。"}
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
    return response.choices[0].message.content or "未收到模型回答。"

def run_agent(question: str) -> dict:
    state = create_agent_state(question)
    collection = get_collection()
    decision = decide_initial_action(question)

    if decision["action"] == "direct_answer":
        state["route"] = "direct_answer"
        state["answer"] = answer_directly(question)
        state["trace"].append("direct_answer")
        return state

    while state["search_attempts"] < MAX_SEARCH_ATTEMPTS:

        state["route"] = "search"
        state["trace"].append(f"search:{state['search_query']}")
        state["results"] = search_knowledge(collection, state["search_query"])
        state["search_attempts"] += 1
        if results_are_sufficient(state["results"]):
            state["route"] = "search_with_results"
            state["answer"] = answer_with_rag(state["question"], state["results"])
            return state
        if state["search_attempts"] >= MAX_SEARCH_ATTEMPTS:
            state["route"] = "no_evidence"
            state["answer"] = "资料中未找到依据。"
            return state
        state["route"] = "rewrite_query"
        # 改写查询前
        state["trace"].append("rewrite_query")
        state["search_query"] = rewrite_query(state["question"], state["search_query"])
        state["trace"].append(f"new_query:{state['search_query']}")
    return state






def rewrite_query(
    original_question: str,
    previous_query: str,
) -> str:
    prompt = f"""
    你是知识库检索查询改写器。

    将用户问题改写成更适合检索本地 Agent 学习资料的关键词。
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

    return new_query or previous_query

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
