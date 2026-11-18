# Running models with vLLM (H200)

The runner talks to a vLLM server through its OpenAI-compatible API (`--model_type VLLM`)
and can send many cases at once (`--num_workers`). Results go to the same folders and use
the same evaluator and scores as before.

## 1. Start the server (GPU node)

Llama 3.1 is gated: accept the license on its Hugging Face page, then `export HF_TOKEN=...`.
vLLM pins its own torch version, so install it in its own environment (`pip install vllm`).

```sh
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --generation-config vllm --max-model-len 16384 --seed 0 --port 8000
```

- `--generation-config vllm`: ignore the model's `generation_config.json` sampling defaults.
  The runner also sends every sampling value explicitly, so this is a second safeguard.
- `--max-model-len 16384`: the longest prompt in the repo is about 15k characters (~4–5k tokens),
  plus 512 output tokens and the step-2 extension.
- `VLLM_BATCH_INVARIANT=1` (beta in vLLM, may be slower): makes outputs independent of how
  requests are batched, so `--num_workers 1` and `--num_workers 32` give identical results.
  Without it, greedy outputs can differ slightly between runs. **Write down whether it was on**:
  the client cannot see the server's environment.

Check it is up:
```sh
curl -s localhost:8000/v1/models    # lists the model id to pass as --model_name
curl -s localhost:8000/version
```
If the server runs in a Kubernetes pod, either run the runner in the same pod, use
`kubectl port-forward pod/<pod> 8000:8000`, or pass `--vllm_base_url http://<host>:8000/v1`.

## 2. Smoke test (5 + 5 cases)

```sh
export PYTHONPATH=.
python3 src/evaluate_prompted_agent.py --model_type VLLM --model_name meta-llama/Llama-3.1-8B-Instruct \
  --setting base --prompt_type InjecAgent --only_first_step \
  --data_dir data/channels/pdf/pymupdf/white --num_workers 16 --limit 5
python3 src/summarize_results.py \
  "results/prompted_VLLM_meta-llama/Llama-3.1-8B-Instruct_InjecAgent_channels-pdf-pymupdf-white_limit5"
```
`--limit` writes to a separate `_limitN` folder, so it never overwrites a full run. Check:
- `0 failed` on the `[dh]` and `[ds]` lines;
- `finish_reason` is mostly `stop` (`length` means the output hit the 512-token cap);
- the maximum prompt tokens are well under 16384;
- open 2–3 rows and read `output`: Thought / Action / Action Input, or a Final Answer.

## 3. Full cell, then the PDF grid

Drop `--limit`. For an 8B model on one H200, start with `--num_workers 32`; the `[dh]`/`[ds]`
lines print cases/s, so you can raise it if the GPU is not saturated.

```sh
for reader in pymupdf ocr; do
  for hiding in clean visible white tiny_1pt small_4pt; do
    python3 src/evaluate_prompted_agent.py --model_type VLLM --model_name meta-llama/Llama-3.1-8B-Instruct \
      --setting base --prompt_type InjecAgent --only_first_step \
      --data_dir data/channels/pdf/$reader/$hiding --num_workers 32
  done
done
```

Each run writes `run_manifest.json` next to its results. It records the parameters, the git
commit, the vLLM version, `max_model_len`, the exact sampling values sent, per-file counts
(written / failed / from cache), timing and scores.

## Options

| Flag | Default | Notes |
|---|---|---|
| `--num_workers` | 1 | Concurrent requests. Refused (error) for local GPU models (`HFChat`, `Llama`). |
| `--limit N` | off | First N cases per file; separate output folder. |
| `--vllm_base_url` | `$VLLM_BASE_URL` or `http://localhost:8000/v1` | |
| `--max_tokens` | 512 | Same as `HFChat`. |
| `--temperature` | 0 | Greedy. With `> 0`, also set `--seed`. |
| `--seed` | none | Per-request seed. |
| `--stop_at_observation` | off | Stops at `\nObservation:`. Usually fewer invalid outputs, but it is a change to InjecAgent's protocol: report it if used. |
| `--chat_template_kwargs` | none | JSON for the chat template, e.g. `'{"enable_thinking": false}'` for Qwen3. |
| `--use_cache` | off | Reuse outputs already in the results folder; re-runs only missing/failed cases. |

## What is (and is not) identical to HFChat

- Same messages ([system, user]), the model's own chat template, greedy decoding, 512 new
  tokens, no stop sequence.
- `VLLM` pins `repetition_penalty=1.0`, `top_k=-1` and `min_p=0`. `HFChat` (transformers) inherits
  any `repetition_penalty` from the model's `generation_config.json`. Llama 3.1 sets none, so the
  two match for Llama. Some other models do set one (check their config).
- Numerics differ between the two engines, so a few greedy outputs may differ. Don't mix
  `HFChat` and `VLLM` results for the same model in one comparison.

## Known issues

- **Failed cases are counted, not hidden.** A request that still fails after 3 retries (e.g. a
  prompt that is too long) is left out of the results file, as before. The run now prints
  `WARNING: N case(s) failed` and records it in the manifest. Fix the cause, then re-run with
  `--use_cache` to fill in only those cases.
- **Data stealing step 2** calls OpenAI `gpt-4-0613` for any attacker-tool call that is not in
  `data/attacker_simulated_responses.json`. For local-only runs, use `--only_first_step`
  (then no `OPENAI_API_KEY` is needed).
  In parallel runs these calls go through a lock, because they rewrite that file.
- **Thinking models** (e.g. Qwen3) put `<think>` text in the output unless you pass
  `--chat_template_kwargs '{"enable_thinking": false}'` (or serve with a reasoning parser).
  That text breaks the ReAct parser.
- **Chat templates without a system role** (e.g. Gemma 2) reject the request. Those cases show up as failures.
