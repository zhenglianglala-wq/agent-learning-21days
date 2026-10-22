from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
import sys



DATABASE_PATH = Path("data/checkpoints.sqlite")
THREAD_ID = "day9-final-demo"

config = {
    "configurable":{"thread_id": THREAD_ID}
}

class ApprovalState(TypedDict):
    filename: str
    content: str
    decision: str
    save_result: str

def request_confirmation(state: ApprovalState):
    decision = interrupt({
        "message": f"是否写入文件{state['filename']}？",
        "options": ["approve", "reject"]
    })
    return {"decision": decision}

def route_after_confirmation(state: ApprovalState):
    if state["decision"] == "approve":
        return "approve"
    else:
        return "reject"

def save_note(state: ApprovalState):
    target = Path("data/practice") / state["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(state["content"], encoding="utf-8")
    return {"save_result": f"File {state['filename']} saved successfully."}


def main():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(DATABASE_PATH)) as checkpointer:
        builder = StateGraph(ApprovalState)
        builder.add_node("request_confirmation", request_confirmation)
        builder.add_node("save_note", save_note)
        builder.add_edge(START, "request_confirmation")
        builder.add_conditional_edges(
            "request_confirmation",
            route_after_confirmation,
            {
                "approve": "save_note",
                "reject": END
            }
        )
        builder.add_edge("save_note", END)
        graph = builder.compile(checkpointer=checkpointer)
        
        
        action = sys.argv[1] if len(sys.argv) > 1 else "start"
        if action == "start":
            result = graph.invoke({"filename": "day9_note.txt", "content": "Day 9 persistence completed."}, config=config)
            print("暂停结果：", result)
        elif action in {"approve", "reject"}:
            result = graph.invoke(Command(resume=action), config=config)
            print("恢复结果：", result)
        else:
            print("用法：python day9_persistence.py [start|approve|reject]")






"""
            
        saved_state = graph.get_state(config)
        if saved_state.values:
                print(f"Loaded state from checkpoint: ", saved_state.values)
            else:
                result = graph.invoke({"count": 0}, config=config)
                print("First invocation result: ", result)
"""
        








if __name__ == "__main__":
    main()