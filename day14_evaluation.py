import json
from pathlib import Path
import time
from day13_agentic_rag import run_agent
from datetime import datetime




EVAL_CASES_FILE = Path("data/eval_cases.json")
RESULTS_FILE = Path("logs/day14_eval_results.jsonl")


def load_eval_cases() -> list[dict]:
    if not EVAL_CASES_FILE.exists():
        raise FileNotFoundError(f"Evaluation cases file not found: {EVAL_CASES_FILE}")
    text = EVAL_CASES_FILE.read_text(encoding="utf-8")
    cases = json.loads(text)
    if not cases:
        raise ValueError("No evaluation cases found in the file.")
    return cases


def evaluate_case(case: dict) -> dict:
    start = time.perf_counter()
    result = run_agent(case["question"])
    duration_seconds = time.perf_counter() - start
    passed = result["route"] ==case["expected_route"]
    source_found = has_sources(result["answer"], result["results"])
    source_passed = source_found ==case["expect_sources"]
    keyword_passed = contains_expected_keywords(
        result["answer"],
        case["expected_keywords"],
    )
    usage_records = result["usage_records"]
    usage_complete = bool(usage_records) and all(
        item.get("total_tokens") is not None for item in usage_records
    )
    total_tokens = sum(item["total_tokens"] for item in usage_records) if usage_complete else None
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_route": case["expected_route"],
        "actual_route": result["route"],
        "passed": passed,
        "duration_seconds": round(duration_seconds, 2),
        "search_attempts": result["search_attempts"],
        "answer": result["answer"],
        "source_passed": source_passed,
        "keyword_passed": keyword_passed,
        "retrieved_results": result["results"],
        "final_query": result["search_query"],
        "trace": result["trace"],
        "planning_usage": result["planning_usage"],
        "rewrite_usage": result["rewrite_usage"],
        "answer_usage": result["answer_usage"],
        "usage_records": usage_records,
        "usage_complete": usage_complete,
        "total_tokens": total_tokens,

    }



def has_sources(answer: str , results: list[dict]) -> bool:
    for item in results:
        source_id = f"{item['filename']}#chunk-{item['chunk_id']}"
        if source_id in answer:
            return True
    return False


def contains_expected_keywords(answer: str, keywords: list[str]) -> bool:
    return all(keyword.lower() in answer.lower() for keyword in keywords)


def evaluate_all_cases(cases: list[dict]) -> list[dict]:
    records = []

    for index, case in enumerate(cases, start=1):
        print(f"正在评估 {index}/{len(cases)}：{case['id']}")

        try:
            record = evaluate_case(case)
        except Exception as error:
            record = {
                "id": case["id"],
                "question": case["question"],
                "passed": False,
                "error": str(error),
            }

        records.append(record)

        print(
            f"  路径：{record.get('actual_route')} "
            f"通过：{record['passed']} "
            f"耗时：{record.get('duration_seconds')} 秒"
        )

    return records

def summarize_results(records: list[dict]) -> None:
    total = len(records)
    passed_count = sum(1 for record in records if record["passed"])

    durations = [
        record["duration_seconds"]
        for record in records
        if "duration_seconds" in record
    ]
    search_counts = [
        record["search_attempts"]
        for record in records
        if "search_attempts" in record
    ]

    print(f"案例总数：{total}")
    print(f"路径通过数：{passed_count}")
    print(f"路径通过率：{passed_count / total:.1%}")

    if durations:
        print(f"平均耗时：{sum(durations) / len(durations):.2f} 秒")
    if search_counts:
        average_searches = sum(search_counts) / len(search_counts)
        print(f"平均搜索次数：{average_searches:.2f}")
    token_totals = [record["total_tokens"] for record in records if record.get("total_tokens") is not None]
    if token_totals:
        print(f"完整用量记录：{len(token_totals)}/{total} 条")
        print(f"完整记录 token 总数：{sum(token_totals)}")
        print(f"完整记录平均 token 数：{sum(token_totals) / len(token_totals):.2f}")

    source_records = [
        record for record in records
        if "source_passed" in record
    ]
    keyword_records = [
        record for record in records
        if "keyword_passed" in record
    ]

    if source_records:

        source_passed_count = sum(
            record["source_passed"] for record in source_records
        )
        source_rate = source_passed_count / len(source_records)
        print(f"来源格式检查通过率：{source_rate:.1%}")

    if keyword_records:
        keyword_passed_count = sum(
            record["keyword_passed"] for record in keyword_records
        )
        keyword_rate = keyword_passed_count / len(keyword_records)
        print(f"关键词检查通过率：{keyword_rate:.1%}")

    failed_records = [
        record for record in records
        if not record["passed"]
    ]

    print(f"\n路径失败案例数：{len(failed_records)}")

    for record in failed_records:
        print(f"\n案例 ID：{record['id']}")
        print(f"问题：{record['question']}")
        print(f"预期路径：{record.get('expected_route', '未记录')}")
        print(f"实际路径：{record.get('actual_route', '执行出错')}")
        print(f"错误信息：{record.get('error', '无')}")



def save_results(records: list[dict]) -> None:
    runtime = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"logs/day14_eval_{runtime}.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", encoding="utf-8") as file:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            file.write(line + "\n")
    print(f"评估结果已保存到 {results_file}")

def load_results(file_path: Path) -> list[dict]:
    records = []
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():  # Skip empty lines
                record = json.loads(line)
                records.append(record)
    return records


def main():
    # 让用户选择要执行的操作
    print("1. 运行新的评估")
    print("2. 查看已有评估结果")
    choice = input("请输入 1 或 2：").strip()

    if choice == "1":
        # 读取评估案例，然后逐条调用 Agent
        cases = load_eval_cases()
        records = evaluate_all_cases(cases)

        # 显示统计结果，并保存本次评估记录
        summarize_results(records)
        save_results(records)

    elif choice == "2":
        # 输入已有的 JSONL 文件路径
        file_name = input("请输入结果文件路径：").strip()
        saved_file = Path(file_name)

        # 读取并统计已有结果，不调用模型
        records = load_results(saved_file)
        summarize_results(records)

    else:
        print("输入无效，请输入 1 或 2。")


# 只有直接运行这个文件时，才启动 main()
if __name__ == "__main__":
    main()
