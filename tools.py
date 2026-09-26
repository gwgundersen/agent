from pathlib import Path


def list_files(path="."):
    return [str(p) for p in Path(path).iterdir()]


def read_file(path):
    return Path(path).read_text()


def write_file(path, contents):
    Path(path).write_text(contents)
    return "ok"


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
}


def execute_tool(call: str) -> str:
    """
    """
    if call["tool"] == "list_files":
        return list_files(call["path"])
    elif call["tool"] == "read_file":
        return read_file(call["path"])
    elif call["tool"] == "write_file":
        return write_file(call["path"], call["contents"])
    else:
        raise NotImplementedError()
