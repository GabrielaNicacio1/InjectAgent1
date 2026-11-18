import os
import sys
import json
import time
import platform
import subprocess
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.output_parsing import evaluate_output_prompted
from src.models import MODELS

# get_simulated_attacker_tool_response() rewrites data/attacker_simulated_responses.json,
# so parallel workers must not run it at the same time.
_SIMULATOR_LOCK = threading.Lock()


def predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=False):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)

    input = item['User Instruction']

    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return output


def _call_info(model_class):
    """finish_reason / token counts of the call just made (models that report them)."""
    return model_class.last_call_info() if hasattr(model_class, 'last_call_info') else None


def process_item(item, attack, params, system_prompt, user_prompt, model_class, tool_dict,
                 existing_data_step1, existing_data_step2):
    """One test case: step 1, evaluation, and (data stealing) step 2.
    Same logic as the original sequential loop; returns the item with results added."""
    if params['use_cache'] and (item['User Instruction'], item['Tool Response']) in existing_data_step1:
        output = existing_data_step1[(item['User Instruction'], item['Tool Response'])]
        item['_cached'] = True
    else:
        output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict)
        info = _call_info(model_class)
        if info is not None:
            item['call_info'] = info
    item['output'] = output

    parsed_output = evaluate_output_prompted(item['output'], item['Attacker Tools'][0], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
    item['eval'] = parsed_output['eval']
    if item['eval'] == 'invalid':
        item['invalid_reason'] = parsed_output['invalid_reason']

    if not params['only_first_step'] and item['eval'] == 'succ' and attack == 'ds':
        item["Attacker Tool Parameters"] = parsed_output["Attacker Tool Parameters"]
        item["Attacker Tool Thought"] = parsed_output["Attacker Tool Thought"]
        with _SIMULATOR_LOCK:
            item["Attacker Tool Response"] = get_simulated_attacker_tool_response(item['Attacker Tools'][0], item["Attacker Tool Parameters"], tool_dict)

        if params['use_cache'] and (item['User Instruction'], item['Tool Response'], item['output']) in existing_data_step2:
            output = existing_data_step2[(item['User Instruction'], item['Tool Response'], item['output'])]
        else:
            output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=True)
            info = _call_info(model_class)
            if info is not None:
                item['call_info Step 2'] = info

        item['output Step 2'] = output
        parsed_output = evaluate_output_prompted(output, item['Attacker Tools'][1], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
        item['eval Step 2'] = parsed_output['eval']
        if item['eval Step 2'] == 'invalid':
            item['invalid_reason  Step 2'] = parsed_output['invalid_reason']
    return item


def _safe(fn, item):
    try:
        return fn(item), None
    except Exception as e:
        return item, e


def run_items(data, fn, num_workers):
    """Yield (item, error) in input order. num_workers > 1 runs cases concurrently;
    results are still yielded (and therefore written) in the original order."""
    if num_workers <= 1:
        for item in tqdm(data):
            yield _safe(fn, item)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            yield from tqdm(ex.map(lambda it: _safe(fn, it), data), total=len(data))


def _git_commit():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout.strip())
        return commit + ('-dirty' if dirty else '')
    except Exception:
        return None


def main(params):
    print(params)
    num_workers = params.get('num_workers') or 1
    model_cls = MODELS[params['model_type']]
    if num_workers > 1 and not getattr(model_cls, 'supports_concurrency', False):
        raise ValueError(f"--num_workers {num_workers} needs a server-backed model (VLLM, GPT, Claude, TogetherAI); "
                         f"'{params['model_type']}' runs on the local GPU and is not thread-safe. Use --num_workers 1.")
    model_class = model_cls(params)
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    output_dir = f"./results/prompted_{params['model_type']}_{params['model_name']}_{params['prompt_type']}"
    data_dir = params.get('data_dir') or './data'
    # Separate output folder per data folder, so channels/conditions never share a cache.
    data_tag = os.path.relpath(os.path.normpath(data_dir), 'data').replace(os.sep, '-')
    if data_tag != '.':
        output_dir += f"_{data_tag}"
    limit = params.get('limit')
    if limit:
        output_dir += f"_limit{limit}"  # never overwrite a full run with a smoke test
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tool_dict = get_tool_dict()
    run_start = time.time()
    manifest = {
        'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'params': params, 'git_commit': _git_commit(), 'python': platform.python_version(),
        'server_info': getattr(model_class, 'server_info', None),
        'request_args': getattr(model_class, 'request_args', None),
        'extra_body': getattr(model_class, 'extra_body', None),
        'files': {},
    }

    output_files = {}
    for attack in ['dh', 'ds']:
        file_name = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        output_files[attack] = output_file

        with open(test_case_file, 'r') as f:
            data = json.load(f)
        if limit:
            data = data[:limit]

        existing_data_step1 = {}
        existing_data_step2 = {}
        if params['use_cache']:
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f.readlines():
                        try:
                            tmp = json.loads(line)
                            if type(tmp['output']) == str and len(tmp['output']):
                                existing_data_step1[(tmp['User Instruction'], tmp['Tool Response'])] = tmp['output']
                            if type(tmp['output Step 2']) == str and len(tmp['output Step 2']):
                                existing_data_step2[(tmp['User Instruction'], tmp['Tool Response'], tmp['output'])] = tmp['output Step 2']
                        except:
                            continue
                print(f"Got {len(existing_data_step1)} existing outputs for the first step and {len(existing_data_step2)} existing outputs for the second step")

        if not params['only_get_score']:
            worker = lambda item: process_item(item, attack, params, system_prompt, user_prompt, model_class,
                                               tool_dict, existing_data_step1, existing_data_step2)
            n_written = n_failed = n_cached = 0
            t0 = time.time()
            with open(output_file, 'w') as f:
                for item, error in run_items(data, worker, num_workers):
                    if error is not None:
                        n_failed += 1
                        print(f"An error occurred: {error} in {output_file}")
                        continue
                    n_cached += bool(item.pop('_cached', False))
                    f.write(json.dumps(item)+'\n')
                    f.flush()
                    n_written += 1
            elapsed = time.time() - t0
            manifest['files'][attack] = {'test_case_file': test_case_file, 'n_cases': len(data), 'n_written': n_written,
                                         'n_failed': n_failed, 'n_from_cache': n_cached, 'elapsed_s': round(elapsed, 2)}
            print(f"[{attack}] {n_written}/{len(data)} written, {n_failed} failed, {n_cached} from cache, "
                  f"{elapsed:.1f}s ({len(data)/elapsed if elapsed else 0:.2f} cases/s, {num_workers} workers)")
            if n_failed:
                print(f"WARNING: {n_failed} case(s) failed and are missing from {output_file}. "
                      f"Re-run with --use_cache to retry only those.")

    scores = get_score(output_files)
    print(json.dumps(scores, indent=True))
    if not params['only_get_score']:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        manifest['elapsed_s'] = round(time.time() - run_start, 2)
        manifest['scores'] = scores
        with open(os.path.join(output_dir, 'run_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=1, default=str)
    return scores

if __name__ == "__main__":
    params = parse_arguments(agent_type="prompted")

    main(params)
