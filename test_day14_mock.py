from unittest.mock import patch
from day14_evaluation import load_eval_cases, evaluate_all_cases
from day13_agentic_rag import run_agent

# 取前两条案例
cases = load_eval_cases()[:2]

# 模拟第一条抛出异常，第二条返回一个正常结果
mock_results = [
    TimeoutError("模拟工具超时"),
    {
        "route": "search_with_results",
        "answer": "向量用于相似检索。来源：17_embedding.md#chunk-0",
        "results": [{"filename": "17_embedding.md", "chunk_id": 0}],
        "search_attempts": 1,
        "search_query": "Embedding 在 RAG 中有什么作用？",
        "trace": ["search:Embedding 在 RAG 中有什么作用？"],
        "planning_usage": None,
        "rewrite_usage": None,
        "answer_usage": None,
        "usage_records": [],
    },
]

# 依次使用上面两个模拟结果替代真实的 run_agent()
with patch("day14_evaluation.run_agent", side_effect=mock_results):
    records = evaluate_all_cases(cases)

print(records)


from day13_agentic_rag import run_agent

with patch("day13_agentic_rag.get_collection", return_value=None):
    with patch("day13_agentic_rag.decide_initial_action", return_value={"action": "search"}):
        with patch("day13_agentic_rag.search_knowledge", side_effect=TimeoutError("模拟检索超时")):
            try:
                run_agent("RAG 是什么？")
            except TimeoutError as error:
                print(f"Agent 抛出了异常：{error}")
