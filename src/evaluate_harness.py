"""Run InjecAgent test cases through one harness and one local model.

Example (pilot: 20 cases per attack type, 3 samples each):
    export PYTHONPATH=.
    python3 src/evaluate_harness.py --harness S3 \
        --model_name Qwen/Qwen2.5-7B-Instruct --setting base \
        --temperature 0.7 --n_samples 3 --limit 20

Outputs go to results/harness_<H>_<model>_<setting>_T<temp>/ as JSONL,
one line per (case, sample), and InjecAgent's usual scores are printed.
"""
import argparse
import json
import os

from tqdm import tqdm

from src.harnesses import HARNESSES
from src.utils import get_tool_dict, get_score

SIMULATED = "./data/attacker_simulated_responses.json"
# Used when a model asks for attacker-tool parameters that are not in the
# cached GPT-4 responses. The original code would call OpenAI here; we stay
# local-only and record that the fallback was used.
FALLBACK_RESPONSE = json.dumps({"status": "success", "result": "Request completed."})


def simulated_attacker_response(tool, params, cache):
    key = f"({tool},{params})"
    if key in cache:
        return cache[key], False
    return FALLBACK_RESPONSE, True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--harness", choices=sorted(HARNESSES), required=True)
    p.add_argument("--model_name", required=True, help="Hugging Face id or local path")
    p.add_argument("--setting", choices=["base", "enhanced"], default="base")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n_samples", type=int, default=1, help="runs per case (seeds 0..n-1)")
    p.add_argument("--limit", type=int, default=None, help="first N cases per attack type")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--only_first_step", action="store_true")
    p.add_argument("--log_prompts", type=int, default=3, help="save the rendered prompt for the first N cases")
    return p.parse_args()


def main(a):
    from src.local_models import LocalChatModel  # imported here so --help works without torch

    harness = HARNESSES[a.harness]
    model = LocalChatModel(a.model_name, a.temperature, a.max_new_tokens)
    tool_dict = get_tool_dict()
    gpt_tools = get_tool_dict(gpt_format=True)
    with open(SIMULATED) as f:
        cache = json.load(f)

    short = a.model_name.rstrip("/").split("/")[-1]
    out_dir = f"./results/harness_{a.harness}_{short}_{a.setting}_T{a.temperature}"
    os.makedirs(out_dir, exist_ok=True)
    output_files = {}

    for attack in ["dh", "ds"]:
        name = f"test_cases_{attack}_{a.setting}.json"
        with open(os.path.join("./data", name)) as f:
            data = json.load(f)[: a.limit]
        output_files[attack] = os.path.join(out_dir, name)

        with open(output_files[attack], "w") as fout:
            for case_id, base_item in enumerate(tqdm(data, desc=attack)):
                for sample in range(a.n_samples):
                    item = dict(base_item, case_id=case_id, sample=sample, seed=sample,
                                harness=a.harness, model=a.model_name, temperature=a.temperature)
                    messages, tools = harness.build(item, tool_dict, gpt_tools)
                    if case_id < a.log_prompts and sample == 0:
                        item["rendered_prompt"] = model.render(messages, tools)
                    item["output"] = model.generate(messages, tools, seed=sample)
                    parsed = harness.evaluate(item["output"], item, item["Attacker Tools"][0])
                    item["eval"] = parsed["eval"]
                    item["invalid_reason"] = parsed.get("invalid_reason")
                    item["tool_calls"] = parsed.get("tool_calls")

                    if attack == "ds" and item["eval"] == "succ" and not a.only_first_step:
                        params = parsed["Attacker Tool Parameters"]
                        item["Attacker Tool Parameters"] = params
                        item["Attacker Tool Thought"] = parsed.get("Attacker Tool Thought", "")
                        resp, used_fallback = simulated_attacker_response(item["Attacker Tools"][0], params, cache)
                        item["Attacker Tool Response"] = resp
                        item["fallback_response"] = used_fallback
                        messages, tools = harness.build(item, tool_dict, gpt_tools, step_2=True)
                        item["output Step 2"] = model.generate(messages, tools, seed=sample)
                        parsed2 = harness.evaluate(item["output Step 2"], item, item["Attacker Tools"][1])
                        item["eval Step 2"] = parsed2["eval"]
                        item["invalid_reason Step 2"] = parsed2.get("invalid_reason")
                    fout.write(json.dumps(item) + "\n")

    print(json.dumps(get_score(output_files), indent=1))


if __name__ == "__main__":
    main(parse_args())
