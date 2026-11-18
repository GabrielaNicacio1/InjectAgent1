import os
import json, struct, math
from src.utils import get_response_text
import time


class BaseModel:
    # True only for models behind a server/API whose client is thread-safe.
    # The runner refuses --num_workers > 1 for anything else (local GPU models).
    supports_concurrency = False

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
    supports_concurrency = True

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
    supports_concurrency = True

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
    
class TogetherAIModel(BaseModel):
    supports_concurrency = True

    def __init__(self, params):
        super().__init__()

        import together  # imported here so the other models work without the together package
        from src.prompts.prompt_template import PROMPT_TEMPLATE

        self.together = together
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
                completion = self.together.Complete.create(
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
        )
        self.gen_kwargs = dict(
            max_new_tokens=512,
            do_sample=False, # temperature=0
            return_full_text = False,
            add_special_tokens=False,
            #eos_token_id=tokenizer.eos_token_id, lets model use own stopped token list
        )

    def prepare_input(self, sys_prompt, user_prompt_filled, tools=None):
        # Was a hard-coded Llama 2 string. Now each model's own template formats it.
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_filled},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

    def prepare_messages(self, messages, tools=None):
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

    def call_model(self, model_input):
        output = self.pipeline(model_input, **self.gen_kwargs)
        return output[0]["generated_text"].strip()
    
class VLLMModel(BaseModel):
    """A model served by vLLM's OpenAI-compatible server (`vllm serve <model>`).

    Mirrors HFChatModel so the two are comparable: [system, user] chat messages,
    the model's own chat template (applied by the server), greedy decoding,
    512 new tokens, no stop sequence. Every sampling parameter is sent on every
    request, so server-side defaults (the model's generation_config.json) never
    apply. Start the server with `--generation-config vllm` as well.

    Thread-safe: the runner may call it from many threads (--num_workers).
    """
    supports_concurrency = True
    STOP_AT_OBSERVATION = ["\nObservation:"]

    def __init__(self, params):
        super().__init__()
        import threading
        from openai import OpenAI

        self.params = params
        self.base_url = (params.get('vllm_base_url') or os.environ.get('VLLM_BASE_URL')
                         or 'http://localhost:8000/v1').rstrip('/')
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get('VLLM_API_KEY', 'EMPTY'),
            timeout=params.get('request_timeout') or 600,
            max_retries=params.get('max_retries', 3),  # connection errors, 408/409/429/5xx, with backoff
        )
        temperature = params.get('temperature')
        self.request_args = {
            'temperature': 0.0 if temperature is None else float(temperature),
            'top_p': 1.0,
            'max_tokens': params.get('max_tokens') or 512,
        }
        if params.get('seed') is not None:
            self.request_args['seed'] = int(params['seed'])
        if params.get('stop_at_observation'):
            self.request_args['stop'] = list(self.STOP_AT_OBSERVATION)
        # vLLM-specific sampling fields, pinned to neutral values (no penalty, no top-k/min-p).
        self.extra_body = {'top_k': -1, 'min_p': 0.0, 'repetition_penalty': 1.0}
        ctk = params.get('chat_template_kwargs')
        if ctk:
            self.extra_body['chat_template_kwargs'] = json.loads(ctk) if isinstance(ctk, str) else ctk
        self._local = threading.local()
        self.server_info = self._check_server()

    def _check_server(self):
        served = self.client.models.list().data
        ids = [m.id for m in served]
        if self.params['model_name'] not in ids:
            raise ValueError(f"vLLM at {self.base_url} serves {ids}, not '{self.params['model_name']}'. "
                             f"Pass the served model name as --model_name.")
        m = next(m for m in served if m.id == self.params['model_name'])
        extra = getattr(m, 'model_extra', None) or {}
        info = {'base_url': self.base_url, 'served_model': m.id, 'root': extra.get('root'),
                'max_model_len': extra.get('max_model_len'), 'vllm_version': None}
        try:
            import urllib.request
            root = self.base_url[:-3] if self.base_url.endswith('/v1') else self.base_url
            with urllib.request.urlopen(root + '/version', timeout=10) as r:
                info['vllm_version'] = json.loads(r.read().decode()).get('version')
        except Exception:
            pass
        return info

    def prepare_input(self, sys_prompt, user_prompt_filled):
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_filled},
        ]

    def call_model(self, model_input):
        self._local.info = None
        completion = self.client.chat.completions.create(
            model=self.params['model_name'],
            messages=model_input,
            extra_body=self.extra_body,
            **self.request_args,
        )
        choice = completion.choices[0]
        usage = completion.usage
        self._local.info = {
            'finish_reason': choice.finish_reason,  # "length" = hit max_tokens
            'prompt_tokens': getattr(usage, 'prompt_tokens', None),
            'completion_tokens': getattr(usage, 'completion_tokens', None),
        }
        return choice.message.content or ""

    def last_call_info(self):
        """finish_reason and token counts of this thread's most recent call."""
        return getattr(self._local, 'info', None)


MODELS = {
    "Claude": ClaudeModel,
    "GPT": GPTModel,
    "Llama": LlamaModel,
    "TogetherAI": TogetherAIModel,
    "HFChat": HFChatModel,
    "VLLM": VLLMModel,
}   