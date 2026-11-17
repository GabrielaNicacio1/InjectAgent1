"""One local model class for every harness.

All harnesses hand this class chat messages, and the model's own chat
template turns them into its prompt format. That way S1-S4 differ only in
the harness, never in how the prompt was formatted for the model.

Use this instead of LlamaModel in models.py: LlamaModel hard-codes the
Llama 2 "[INST] <<SYS>>" format, which is wrong for Llama 3.1.
"""
import torch
import transformers


class LocalChatModel:
    def __init__(self, model_name, temperature=0.0, max_new_tokens=512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    def render(self, messages, tools=None):
        """The exact prompt string the model sees. Log it for a few cases."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(self, messages, tools=None, seed=0):
        prompt = self.render(messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        kwargs = dict(max_new_tokens=self.max_new_tokens,
                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if self.temperature > 0:
            torch.manual_seed(seed)
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=1.0)
        else:
            kwargs.update(do_sample=False)
        out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        # Qwen's <tool_call> tags survive skip_special_tokens; Llama's and
        # Mistral's tool-call markers are dropped, leaving plain JSON, which
        # harnesses.parse_tool_calls() handles.
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
