import json
from pathlib import Path
import sys
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tools --------------------------------------------------------------------------------------------

def list_files(path="."): return [str(p) for p in Path(path).iterdir()]

def read_file(path): return Path(path).read_text()

def write_file(path, contents): Path(path).write_text(contents)

# Model --------------------------------------------------------------------------------------------

class LLM:

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self):
        """Load weights and tokenizer for Qwen large language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, torch_dtype="auto", device_map="auto"
        )

    def __call__(self, prompt: str) -> str:
        """Get Gwen response from prompt."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=32768)
        generated = outputs[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

# Agentic loop -------------------------------------------------------------------------------------

AGENT_PROMPT_TEMPLATE = """
You are an agent that completes the user's task by using tools.

USER TASK: {prompt}

RESPONSE: You must respond with exactly one JSON object and no other text. To call a tool:
  {{"type": "tool", "name": <function_name>, "kwargs": <dict of keyword args>}}
When and only when you have enough information to answer the user's task:
  {{"type": "final", "text": <your answer>}}

<your answer> should be a string which answers the user prompt.

HISTORY: {history}

AVAILABLE TOOLS:
  list_files(path=".")         Returns files and directories directly inside path.
  read_file(path)              Returns the full contents of the file at path.
  write_file(path, contents)   Writes contents to path and returns "ok".

Returns files and directories directly inside path.
"""

def step(prompt: str, model: LLM, tools: dict[str, Callable], history: list[dict]) -> dict:
    """Single agentic "step": feed prompt, tools, and action history into LLM and get response."""
    # Construct agent prompt from user prompt and agent action history
    agent_prompt = AGENT_PROMPT_TEMPLATE.format(prompt=prompt, history=json.dumps(history))
    # Feed prompt into LLM
    raw = model(agent_prompt)
    # Parse agent response into structured output
    resp = json.loads(raw)
    # Take action with available tool or return answer
    if resp["type"] == "tool":
        fn = tools[resp["name"]]
        result = fn(**resp["kwargs"])
        return {"response": resp, "result": result, "status": "continue"}
    else:
        return {"response": resp, "result": resp["text"], "status": "terminate"}

def run_agent(prompt: str, model: LLM, tools: dict, max_actions: int) -> str:
    """Run an 'agent', meaning call an LLM recursively with a set of tools and action history until
    the agent terminates or the max number of actions was taken."""
    history = []
    for i in range(max_actions):
        resp = step(prompt, model, tools, history)
        history.append(resp)
        if resp["status"] != "continue":
            break
    return history

# Main program -------------------------------------------------------------------------------------

# Example usage: uv run python main.py "Summarize the contents of agent-micro.py"
if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:])
    model = LLM()
    tools =  {"list_files": list_files, "read_file": read_file, "write_file": write_file}
    actions = run_agent(prompt, model, tools, max_actions=10)
    for node in actions:
        print(f"{'=' * 100}\n{node}")
