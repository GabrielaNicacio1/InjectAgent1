import os
import json, struct, math
from src.utils import get_response_text
import time



# ---- 2) Parameter-count guard ----
'''MAX_PARAMS_B = float(os.environ.get("MAX_PARAMS_B", 7.7))   # billions

def _count_params_b(name):
    """Return parameter count in billions, or None if it can't be determined."""
    if os.path.isdir(name):
        # Local checkpoint: sum tensor shapes from .safetensors headers (dtype-independent)
        total, found = 0, False
        for f in os.listdir(name):
            if not f.endswith(".safetensors"):
                continue
            found = True
            with open(os.path.join(name, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            total += sum(math.prod(v["shape"]) for k, v in header.items()
                         if k != "__metadata__")
        return total / 1e9 if found else None
    # Hub repo: read parameter count from metadata, no weights downloaded
    from huggingface_hub import HfApi
    st = HfApi().model_info(name, expand=["safetensors"]).safetensors
    return st.total / 1e9 if st and st.total else None

def check_model_size(name, max_b=MAX_PARAMS_B):
    b = _count_params_b(name)
    if b is None:
        raise ValueError(f"{name}: cannot determine parameter count; refusing to load.")
    if b > max_b:
        raise ValueError(f"{name}: {b:.1f}B parameters exceeds limit of {max_b}B.")
    print(f"[size check] {name}: {b:.1f}B parameters (limit {max_b}B)")


#####################
'''
class BaseModel:
    def __init__(self):
        self.model = None

    def prepare_input(self, sys_prompt,  user_prompt_filled):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def call_model(self, model_input):
        raise NotImplementedError("This method should be overridden by subclasses.")


####
class HFChatModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        import torch
        import transformers

        name = params['model_name']
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def prepare_input(self, sys_prompt, user_prompt_filled):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_filled},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def call_model(self, model_input):
        inputs = self.tokenizer(
            model_input, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        output = self.model.generate(
            **inputs, max_new_tokens=512, do_sample=False
        )
        # Decode only the newly generated tokens (no prompt echo)
        return self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
    

    ############
class ClaudeModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(
            api_key= os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.params = params
        self.human_prompt = HUMAN_PROMPT
        self.ai_prompt = AI_PROMPT

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"{self.human_prompt} {sys_prompt} {user_prompt_filled}{self.ai_prompt}"
        return model_input

    def call_model(self, model_input):
        completion = self.anthropic.completions.create(
            model=self.params['model_name'],
            max_tokens_to_sample=4096,
            prompt=model_input,
            temperature=0
        )
        return completion.completion
    
class GPTModel(BaseModel):
    def __init__(self, params):
        super().__init__()  
        from openai import OpenAI
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY"),
            organization = os.environ.get("OPENAI_ORGANIZATION")
        )
        self.params = params

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = [
            {"role": "system", "content":sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]
        return model_input

    def call_model(self, model_input):
        completion = self.client.chat.completions.create(
            model=self.params['model_name'],
            messages=model_input,
            temperature=0
        )
        return completion.choices[0].message.content
    
import together   
class TogetherAIModel(BaseModel): 
    def __init__(self, params):
        super().__init__()  
        
        from src.prompts.prompt_template import PROMPT_TEMPLATE
        
        self.params = params
        self.model = self.params['model_name']
        self.prompt = PROMPT_TEMPLATE[self.model]

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = self.prompt.format(sys_prompt = sys_prompt, user_prompt = user_prompt_filled)
        return model_input
    
    def call_model(self, model_input, retries=3, delay=2, max_tokens =512):
        attempt = 0
        while attempt < retries:
            try:
                completion = together.Complete.create(
                    model=self.model,
                    prompt=model_input,
                    max_tokens=max_tokens,
                    temperature=0
                )
                return completion['choices'][0]['text']
            except Exception as e:
                attempt += 1
                max_tokens = max_tokens // 2
                print(f"Attempt {attempt}: An error occurred - {e}")
                if attempt < retries:
                    time.sleep(delay)  # Wait for a few seconds before retrying
                else:
                    return ""
    
class LlamaModel(BaseModel):     
    def __init__(self, params):
        super().__init__() 
        ###########
        #check_model_size(params['model_name']) 
        ########
        import transformers
        import torch
        tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'])
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=params['model_name'],
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_length=4096,
            do_sample=False, # temperature=0
            eos_token_id=tokenizer.eos_token_id,
        )

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_prompt_filled} [/INST]"
        return model_input

    def call_model(self, model_input):
        output = self.pipeline(model_input)
        return get_response_text(output, "[/INST]")
    
MODELS = {
    "Claude": ClaudeModel,
    "GPT": GPTModel,
    "Llama": LlamaModel,
    "TogetherAI": TogetherAIModel,
    "HFChat": HFChatModel
}   