"""Which harness works with which model? Run this before any GPU time.

It loads only each model's tokenizer (no weights) and tries to render one
real InjecAgent case through every harness with the model's chat template.
A harness is compatible with a model when its messages render cleanly; a
template that rejects a system message, tool schemas or tool results fails here.

    export PYTHONPATH=.
    huggingface-cli login            # Llama 3.1 is gated: accept its licence first
    python3 scripts/check_compat.py
    python3 scripts/check_compat.py --models Qwen/Qwen2.5-7B-Instruct --show S3

Rendering cleanly is necessary, not sufficient: the pilot still has to show
that the model emits a tool call our parser can read (check Valid Rate).
"""
import argparse
import json

import transformers

from src.harnesses import HARNESSES
from src.utils import get_tool_dict

DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--show", default=None, help="print the full rendered prompt for this harness")
    a = p.parse_args()

    tool_dict = get_tool_dict()
    gpt_tools = get_tool_dict(gpt_format=True)
    with open("data/test_cases_ds_base.json") as f:
        item = json.load(f)[0]
    # Fake a step-2 state so the two-tool-call path is checked too.
    item = dict(item, **{"Attacker Tool Parameters": "{}", "Attacker Tool Thought": "",
                         "Attacker Tool Response": "{\"status\": \"ok\"}"})

    rows = []
    for name in a.models:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        template = tok.chat_template or ""
        if isinstance(template, dict):          # some repos ship several named templates
            template = " ".join(template.values())
        print(f"\n== {name}")
        print(f"   template mentions tools: {'tools' in template}  | system role: {'system' in template}")
        for h_name, h in HARNESSES.items():
            for step_2 in (False, True):
                try:
                    msgs, tools = h.build(item, tool_dict, gpt_tools, step_2=step_2)
                    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
                    n_tok = len(tok(text, add_special_tokens=False).input_ids)
                    status = f"OK ({n_tok} tokens)"
                    if a.show == h_name and not step_2:
                        print(text)
                except Exception as e:  # noqa: BLE001 - we want the template's message
                    status = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
                rows.append((name.split("/")[-1], h_name, "step 2" if step_2 else "step 1", status))
                print(f"   {h_name:4} {'step 2' if step_2 else 'step 1'}: {status}")

    print("\nSummary (model, harness, step, status)")
    for r in rows:
        print("  " + " | ".join(r))


if __name__ == "__main__":
    main()
