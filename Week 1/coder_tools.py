"""
Coder agent's tools: read/write files and execute Python code.

SAFETY FLAG -- read before running on anything you care about:
`execute_code` runs arbitrary Python via exec() with NO sandboxing, per
your choice of "simplest, less safe." This means any code the LLM decides
to generate -- including code shaped by content the researcher agent
pulled in from Wikipedia -- runs with your full user permissions. Fine
for a controlled resume demo you're driving yourself. NOT fine to leave
running unattended or point at untrusted goals. A safer follow-up
(flagged, not built here) would be running generated code in a subprocess
with reduced permissions, or a container (Docker), instead of exec()
in-process.

File operations ARE scoped to a workspace directory (`_safe_path` blocks
path traversal like "../../etc/passwd"), but that only protects the
filesystem -- it does nothing to sandbox exec() itself.
"""
import io
import contextlib
import os

WORKSPACE_DIR = os.environ.get("AGENT_WORKSPACE", "./agent_workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


def _safe_path(path: str) -> str:
    """Keep file operations inside WORKSPACE_DIR."""
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not full.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise ValueError(f"Path '{path}' escapes the workspace directory.")
    return full


def read_file(path: str) -> str:
    full = _safe_path(path)
    if not os.path.exists(full):
        return f"File '{path}' does not exist."
    with open(full, "r") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    full = _safe_path(path)
    os.makedirs(os.path.dirname(full) or WORKSPACE_DIR, exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to '{path}'."


def list_directory(path: str = ".") -> list:
    full = _safe_path(path)
    if not os.path.exists(full):
        return []
    return os.listdir(full)


def execute_code(code: str) -> str:
    """
    Run Python code via exec() and capture stdout.
    See module docstring for the safety caveat -- this is intentionally
    unsandboxed per your choice.
    """
    stdout_capture = io.StringIO()
    local_scope = {}
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {"__builtins__": __builtins__}, local_scope)
        output = stdout_capture.getvalue()
        return output if output else "(code ran with no printed output)"
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {e}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file inside the agent workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file inside the agent workspace (overwrites if it exists).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory inside the agent workspace. Defaults to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code and return captured stdout/errors. Use for running scripts, quick calculations, or testing generated code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "execute_code": execute_code,
}
