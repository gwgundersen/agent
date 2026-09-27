import json
from json.decoder import JSONDecodeError
from pathlib import Path
import re
import sys
from typing import Callable

from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------------

def list_files(path="."):
    return [str(p) for p in Path(path).iterdir()]


def read_file(path):
    return Path(path).read_text()


def write_file(path, contents):
    Path(path).write_text(contents)
    return "ok"


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------

class LLM:

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self):
        """
        Load weights and tokenizer for Qwen large language model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )    

    def __call__(self, prompt: str) -> str:
        """
        Get Gwen response from prompt.
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=32768
        )
        generated = outputs[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# --------------------------------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------------------------------

# Max number of agent actions.
MAX_ACTIONS = 10

# Max number of retries if agent does not generate compliant JSON.
MAX_TRIES = 5

AGENT_PROMPT_TEMPLATE = """
You are an agent that completes the user's task by using tools.

USER TASK:
{prompt}

You must respond with exactly one JSON object and no other text.

To call a tool:
{{"type": "tool", "name": <function_name>, "kwargs": <dict of keyword args>}}

When and only when you have enough information to answer the user's task:
{{"type": "final", "text": <your answer>}}

<your answer> should be a string which answers the user prompt.

HISTORY:
  {interaction_history}

AVAILABLE TOOLS:

  list_files(path=".")
  Returns files and directories directly inside path.

  read_file(path)
  Returns the full contents of the file at path.

  write_file(path, contents)
  Writes contents to path and returns "ok".

Returns files and directories directly inside path.
"""

class ParsingError(ValueError):
    pass


def build_prompt(prompt: str, history: list[dict]) -> str:
    """
    Template user response with agentic directions and action history.
    """
    return AGENT_PROMPT_TEMPLATE.format(
        prompt=prompt,
        interaction_history=json.dumps(history)
    )


def parse_response(response: str, tools: dict[str, Callable]) -> dict:
    """
    An LLM response is a sequence of tokens converted into a string. Parse this string into
    structured output. Raises `ParsingError` on a malformed response.
    """
    try:
        obj = json.loads(response)
    except JSONDecodeError as e:
        raise ParsingError("agent response is not well-formed JSON")

    if "type" not in obj:
        raise ParsingError("invalid response object")

    if obj["type"] not in ["tool", "final"]:
        raise ParsingError("invalid action type")

    if obj["type"] == "tool":
        if obj["name"] not in tools.keys():
            raise ParsingError("invalid tool name")
        return obj

    if obj["type"] == "final":
        if not isinstance(obj["text"], str):
            raise ParsingError("final text should be a string")

    return obj


def step(
    prompt: str,
    model: LLM,
    tools: dict[str, Callable],
    history: list[dict]
) -> dict:
    """
    Wrapper for recursive call to `_step` in case single call fails.
    """
    return _step(prompt, model, tools, history, MAX_TRIES)


def _step(
    prompt: str,
    model: LLM,
    tools: dict[str, Callable],
    history: list[dict],
    n_tries: int,
) -> dict:
    """
    Single agentic "step": feed prompt, tools, and action history into LLM and get response.
    """
    try:
        # Construct agent prompt from user prompt and agent action history
        agent_prompt = build_prompt(prompt, history)
        # Feed prompt into LLM
        raw = model(agent_prompt)
        # Parse response into structured output
        resp = parse_response(raw, tools)
    
        # Take action with available tool
        if resp["type"] == "tool":
            fn = tools[resp["name"]]
            result = fn(**resp["kwargs"])
            return {
                "prompt": prompt,
                "response": resp,
                "result": result,
                "status": "continue"
            }

        # Return answer
        else:
            return {
                "prompt": prompt,
                "response": resp,
                "result": resp["text"],
                "status": "terminate"
            }
    except ParsingError as e:
        if n_tries > 1:
            # FIXME: Add failure to history
            _step(prompt, model, tools, history, n_tries - 1)
        else:
            raise ValueError(f"failed to parse agent response in {max_tries} tries")


def run_agent(prompt: str, model: LLM, tools: dict) -> str:
    """
    Run an 'agent', meaning call an LLM recursively with a set of tools and action history until the
    agent terminates or the max number of actions was taken.

    Example usage:
      uv run python main.py "Summarize the contents of agent.py"
    """
    history = []
    for i in range(MAX_ACTIONS):
        resp = step(prompt, model, tools, history)
        if resp["status"] == "continue":
            history.append(resp)
        else:
            history.append(resp)
            break
    return history


# --------------------------------------------------------------------------------------------------
# User input
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:])
    model = LLM()
    tools =  {
        "list_files": list_files,
        "read_file": read_file,
        "write_file": write_file,
    }
    history = run_agent(prompt, model, tools)
    for node in history:
        print("=" * 100)
        print(node)
