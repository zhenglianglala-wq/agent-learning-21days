
from pathlib import Path
from mcp.server import MCPServer



PRACTICE_DIR = Path(__file__).resolve().parent /"data" / "practice"
PRACTICE_DIR.mkdir(parents=True, exist_ok=True)
mcp = MCPServer("Day7FileTools")


def get_safe_path(filename: str) -> Path:
    path = (PRACTICE_DIR / filename).resolve()
    path.relative_to(PRACTICE_DIR.resolve())  # Ensure the path is within the practice directory
    return path


@mcp.tool()
def list_files() -> dict:
    """
    List all files in the practice directory.
    """
    files = [path.name for path in PRACTICE_DIR.iterdir() if path.is_file()]
    return {"files": files, "count": len(files)}


@mcp.tool()
def read_file(filename: str) -> dict:
    """
    Read the content of a file in the practice directory.
    """
    path = get_safe_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    return {"filename": filename, "content": path.read_text(encoding="utf-8")}

@mcp.tool()
def search_files(keyword: str) -> dict:
    """
    Search for files containing the given keyword in the practice directory.
    """
    if not keyword.strip():
        raise ValueError("Keyword must not be empty.")
    needle = keyword.lower()
    matching_files = []
    for path in PRACTICE_DIR.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").lower()
        if needle in path.name.lower() or needle in content:
            matching_files.append(path.relative_to(PRACTICE_DIR).as_posix())
    return {"keyword": keyword, "matching_files": matching_files, "count": len(matching_files)}

@mcp.tool()
def save_note(filename: str, content: str) -> dict:
    """
    Save content to a file in the practice directory.
    """
    path = get_safe_path(filename)
    if not content.strip():
        raise ValueError("Content must not be None.")  
    if path.exists():
        old_content = path.read_text(encoding="utf-8")
        if old_content == content:
            return {"filename": path.name, "status": "unchanged", "characters_written": len(content)}
        status = "updated"
    else:
        status = "created"
    path.write_text(content, encoding="utf-8")
    return {"filename": path.name, "status": status, "characters_written": len(content)}



if __name__ == "__main__":
    mcp.run()
