
import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_DIR = Path(__file__).resolve().parent
SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_DIR / "day7_mcp_server.py")],
    cwd=PROJECT_DIR,
)


async def main() -> None:
    async with Client(SERVER) as client:
        tools_result = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools_result.tools])

        calls = [
            ("list_files", {}
            ),
            (
                "read_file", {"filename": "todo.txt"}
            ),
            (
                "search_files", {"keyword": "Function"}
            )
            ,
            (
                "save_note", {"filename": "day7_note.txt", "content": "finished day7 mcp client test."}
            )
        ]
        for tool_name, arguments in calls:
            
            result = await client.call_tool(tool_name, arguments)
                
            if result.is_error:
                print(f"Error calling {tool_name}: ")
            else:
                print(f"Result of {tool_name}: ")
            
            for block in result.content:
                if hasattr(block, "text"):
                    print(block.text)

        


if __name__ == "__main__":
    asyncio.run(main())
