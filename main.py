import sys

from llm import LLM
from tools import TOOLS

# ------------------------------------------------------------------------------

MAX_STEPS = 10
AGENT_PROMPT = """
    You are an agent. Always output

        TOOL: <tool_name>
        ARGS: {"argument": "<value>"}
    or
        FINAL: <answer>

    You have access to these tools:

    1. list_files(path=".")
       - Returns the files and directories directly inside `path`.
       - Use this when you need to discover what files exist.

    2. read_file(path)
       - Returns the full text contents of the file at `path`.
       - Use this when you need to inspect a file.

    3. write_file(path, contents)
       - Replaces the file at `path` with `contents`.
       - Returns "ok" on success.
       - Use this when you need to create or modify a text file.

    When you want to use a tool, output exactly one tool call in this format:

    TOOL: tool_name
    ARGS: {"argument": "value"}
"""


def run_agent(model: LLM, state: dict) -> None:
    """
    agent =
        LLM
        + loop
        + state/history
        + tools/actions
        + stopping condition
    """
    prompt = f"{AGENT_PROMPT}\n{state['prompt']}"
    for step in range(MAX_STEPS):
        output = model(prompt)
        if output.type == "tool_call":
            output = execute_tool(output.tool_call)
            state += result
        elif output.type == "final":
            return


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:])
    model = LLM()
    run_agent(
        model=model,
        state={
            "prompt": goal,
            "history": [],
        }
    )
