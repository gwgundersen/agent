from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------------------

class LLM:

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    def __init__(self):
        """
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )    

    def __call__(self, prompt: str) -> str:
        """
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=32768
        )
        generated = outputs[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True) x
