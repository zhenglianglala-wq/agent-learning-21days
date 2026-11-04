import asyncio
import json
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4
from copy import deepcopy
from importlib.metadata import version
from typing import Callable
import os

from dotenv import load_dotenv
from langfuse import get_client
from day15_multi_agent import (
    create_state,
    researcher,
    reviewer,
    supervisor,
    writer,
)
from day12_rag import MODEL, client



LOG_DIR = Path(__file__).resolve().parent / "logs"
EVAL_SOURCE_FILE = LOG_DIR / "day14_eval_20261029_170241.jsonl"
FAITHFULNESS_CASE_IDS = (
    "answerable-01",
    "answerable-07",
    "answerable-09",
    "boundary-04",
    "multi-step-01",
)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceLogger:
    def __init__(self, question: str, log_dir: Path= LOG_DIR) -> None:
        self.trace_id = uuid4().hex
        self.question = question
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "day16_trace.jsonl"

    def write_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    @contextmanager
    def span(self, stage: str, inputs: dict[str, Any] | None = None, parent_span_id: str | None = None) -> Iterator[dict[str, Any]]:
        event: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": uuid4().hex,
            "parent_span_id": parent_span_id,
            "question": self.question,
            "stage": stage,
            "started_at": utc_now(),
            "ended_at": None,
            "duration_seconds": None,
            "status": "running",
            "inputs": inputs if inputs is not None else {},
            "outputs": {},
            "usage_records": [],
            "usage_complete": True,
            "total_tokens": 0,
            "error": None,
        }
        started = time.perf_counter()
        try:
            yield event
            event["status"] = "success"
        except BaseException as exc:
            event["status"] = "error"
            event["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            event["ended_at"] = utc_now()
            event["duration_seconds"] = round(
                time.perf_counter() - started,
                4,
            )

            usage_records = event["usage_records"]
            usage_complete = all(
                record.get("total_tokens") is not None
                for record in usage_records
            )

            event["usage_complete"] = usage_complete
            event["total_tokens"] = (
                sum(
                    record["total_tokens"]
                    for record in usage_records
                )
                if usage_complete
                else None
            )

            self.write_event(event)


def run_logged_stage(
    logger: TraceLogger,
    state: dict,
    stage: str,
    operation: Callable[[dict], Any],
    input_keys: tuple[str, ...],
    output_keys: tuple[str, ...],
) -> Any:
    langfuse = get_client()

    inputs = deepcopy({
        key: state[key]
        for key in input_keys
    })

    usage_start = len(state["usage_records"])

    with logger.span(stage, inputs=inputs) as event:
        with langfuse.start_as_current_observation(
            as_type="span",
            name=stage,
            input=inputs,
            metadata={
                "local_span_id": event["span_id"],
                "revision_count": state["revision_count"],
            },
        ) as observation:
            try:
                result = operation(state)

                event["outputs"] = deepcopy({
                    key: state[key]
                    for key in output_keys
                })

                if result is not None:
                    event["outputs"]["return_value"] = result

                observation.update(output=event["outputs"])

                return result

            except BaseException as exc:
                observation.update(
                    level="ERROR",
                    status_message=f"{type(exc).__name__}: {exc}",
                )
                raise

            finally:
                event["usage_records"] = deepcopy(
                    state["usage_records"][usage_start:]
                )

                observation.update(
                    metadata={
                        "usage_records": event["usage_records"],
                    }
                )

def run_observed_workflow(
    question: str,
    logger: TraceLogger,
) -> dict:
    state = create_state(question)

    run_logged_stage(
        logger=logger,
        state=state,
        stage="researcher",
        operation=researcher,
        input_keys=("question",),
        output_keys=("research_results",),
    )

    while True:
        run_logged_stage(
            logger=logger,
            state=state,
            stage="writer",
            operation=writer,
            input_keys=(
                "question",
                "research_results",
                "draft",
                "review",
                "revision_count",
            ),
            output_keys=("draft",),
        )

        if not state["research_results"]:
            state["stop_reason"] = "no_evidence"
            return state

        run_logged_stage(
            logger=logger,
            state=state,
            stage="reviewer",
            operation=reviewer,
            input_keys=(
                "question",
                "research_results",
                "draft",
                "revision_count",
            ),
            output_keys=("review",),
        )

        action = run_logged_stage(
            logger=logger,
            state=state,
            stage="supervisor",
            operation=supervisor,
            input_keys=("review", "revision_count"),
            output_keys=("revision_count",),
        )

        if action in ("finish", "stop"):
            state["stop_reason"] = action
            return state

        if action != "revise":
            raise ValueError(f"未知调度动作：{action}")

        state["revision_count"] += 1

def load_faithfulness_cases(file_path: Path) -> list[dict]:
    records = {}
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record

    missing = [case_id for case_id in FAITHFULNESS_CASE_IDS if case_id not in records]
    if missing:
        raise ValueError(f"日志中缺少案例：{missing}")

    return [records[case_id] for case_id in FAITHFULNESS_CASE_IDS]


async def evaluate_faithfulness_case(case: dict, scorer: Any) -> dict:
    contexts = [
        item["text"]
        for item in case.get("retrieved_results", [])
        if item.get("text")
    ]
    record = {
        "id": case["id"],
        "question": case["question"],
        "answer": case["answer"],
        "retrieved_results": case.get("retrieved_results", []),
        "keyword_passed": case.get("keyword_passed"),
        "judge_model": MODEL,
        "ragas_version": version("ragas"),
        "status": "pending",
        "faithfulness": None,
        "duration_seconds": None,
        "error": None,
        "human_review": {"supported": None, "complete": None, "notes": ""},
    }

    if (
        case.get("actual_route") != "search_with_results"
        or not contexts
        or not case["answer"].strip()
    ):
        record["status"] = "skipped"
        record["skip_reason"] = "没有可评估的检索回答或资料"
        return record

    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            scorer.ascore(
                user_input=case["question"],
                response=case["answer"],
                retrieved_contexts=contexts,
            ),
            timeout=180,
        )
        score = float(result.value)
        if not math.isfinite(score):
            record["status"] = "unscorable"
            record["error"] = {
                "type": "NonFiniteScore",
                "message": "未得到有效分数，可能未提取到可判断的结论。",
            }
        elif not 0 <= score <= 1:
            raise ValueError(f"评分超出范围：{score}")
        else:
            record["status"] = "success"
            record["faithfulness"] = score
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        record["duration_seconds"] = round(time.perf_counter() - started, 2)

    return record


async def run_faithfulness_evaluation() -> None:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    cases = load_faithfulness_cases(EVAL_SOURCE_FILE)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = LOG_DIR / f"day16_faithfulness_{timestamp}.jsonl"
    scores = []

    print("\n读取已有回答进行评估，将调用模型评分。")
    print("来源日志：", EVAL_SOURCE_FILE)
    print("结果位置：", output_path)

    async with AsyncOpenAI(
        api_key=client.api_key,
        base_url=str(client.base_url),
        timeout=60,
        max_retries=1,
    ) as judge_client:
        scorer = Faithfulness(llm=llm_factory(MODEL, client=judge_client))

        for case in cases:
            print(f"\n正在评估：{case['id']} {case['question']}")
            record = await evaluate_faithfulness_case(case, scorer)
            record["source_file"] = str(EVAL_SOURCE_FILE)
            with output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

            print("状态：", record["status"])
            print("忠实度：", record["faithfulness"])
            if record["error"]:
                print("错误：", record["error"])
            if record["status"] == "success":
                scores.append(record["faithfulness"])

    print(f"\n有效评分：{len(scores)}/{len(cases)}")
    if scores:
        print(f"平均忠实度：{sum(scores) / len(scores):.3f}")
    print("结果保存到：", output_path)


def run_tracing() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    required_variables = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    )

    missing = [
        name
        for name in required_variables
        if not os.getenv(name)
    ]

    if missing:
        print("缺少环境变量：", ", ".join(missing))
        return

    question = input("请输入报告主题：").strip()

    if not question:
        print("报告主题不能为空。")
        return

    logger = TraceLogger(question)
    langfuse = get_client()

    print(f"\nTrace ID：{logger.trace_id}")

    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="day16_multi_agent",
            trace_context={"trace_id": logger.trace_id},
            input={"question": question},
        ) as root:
            state = run_observed_workflow(question, logger)

            root.update(
                output={
                    "answer": state["draft"],
                    "review": state["review"],
                    "revision_count": state["revision_count"],
                    "stop_reason": state["stop_reason"],
                }
            )

    except Exception as exc:
        print(f"\n执行失败：{type(exc).__name__}: {exc}")
        print(f"日志位置：{logger.log_path}")
        return

    finally:
        langfuse.flush()

    print("\n最终报告：")
    print(state["draft"])

    print("\n最终审阅意见：")
    print(state["review"])

    print("\n修改次数：", state["revision_count"])
    print("停止原因：", state["stop_reason"])
    print("执行链路：", " -> ".join(state["trace"]))
    print("日志位置：", logger.log_path)
    print("请到 Langfuse 项目中查看对应 Trace ID。")


def main() -> None:
    print("1. 多角色工作流与 Langfuse 调用链")
    print("2. RAGAS 忠实度评估（Day14 五个案例）")
    mode = input("请选择模式（默认 1）：").strip() or "1"

    if mode == "1":
        run_tracing()
    elif mode == "2":
        try:
            asyncio.run(run_faithfulness_evaluation())
        except Exception as exc:
            print(f"\n评估启动或保存失败：{type(exc).__name__}: {exc}")
    else:
        print("无效模式，请输入 1 或 2。")


if __name__ == "__main__":
    main()
