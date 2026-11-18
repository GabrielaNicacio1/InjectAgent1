"""Summarize one results folder: outcomes, why outputs were invalid, truncation, tokens.

    PYTHONPATH=. python3 src/summarize_results.py results/prompted_VLLM_<model>_InjecAgent_<cell>

Use it after a smoke test, and to diagnose high invalid rates.
"""

import json
import os
import sys
from collections import Counter


def summarize(out_dir):
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("test_cases_") and name.endswith(".json"):
            with open(os.path.join(out_dir, name)) as f:
                rows += [dict(json.loads(line), _file=name) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"no result rows in {out_dir}")

    print(f"{out_dir}\n{len(rows)} rows from {sorted({r['_file'] for r in rows})}\n")
    print("eval (step 1):", dict(Counter(r["eval"] for r in rows)))
    inv = Counter(r.get("invalid_reason") for r in rows if r["eval"] == "invalid")
    if inv:
        print("invalid reasons:")
        for reason, n in inv.most_common():
            print(f"  {n:4}  {reason}")
    s2 = [r for r in rows if "eval Step 2" in r]
    if s2:
        print("eval (step 2):", dict(Counter(r["eval Step 2"] for r in s2)))

    infos = [r["call_info"] for r in rows if r.get("call_info")]
    if infos:
        fin = Counter(i["finish_reason"] for i in infos)
        print("finish_reason:", dict(fin), "(length = cut off at max_tokens)")
        pt = [i["prompt_tokens"] for i in infos if i.get("prompt_tokens") is not None]
        ct = [i["completion_tokens"] for i in infos if i.get("completion_tokens") is not None]
        if pt:
            print(f"prompt tokens: min {min(pt)}, mean {sum(pt)/len(pt):.0f}, max {max(pt)}")
        if ct:
            print(f"completion tokens: min {min(ct)}, mean {sum(ct)/len(ct):.0f}, max {max(ct)}")
        trunc_invalid = sum(1 for r in rows if r.get("call_info", {}).get("finish_reason") == "length"
                            and r["eval"] == "invalid")
        if trunc_invalid:
            print(f"{trunc_invalid} invalid outputs were also cut off at max_tokens")
    else:
        print("no call_info (cached outputs, or a model type that does not report it)")

    manifest = os.path.join(out_dir, "run_manifest.json")
    if os.path.exists(manifest):
        m = json.load(open(manifest))
        failed = {k: v["n_failed"] for k, v in m.get("files", {}).items() if v.get("n_failed")}
        print(f"\nmanifest: {m.get('elapsed_s')}s, vLLM {(m.get('server_info') or {}).get('vllm_version')}, "
              f"git {m.get('git_commit')}" + (f", FAILED cases: {failed}" if failed else ""))
        print("scores:", json.dumps(m.get("scores")))


if __name__ == "__main__":
    for d in sys.argv[1:] or ["."]:
        summarize(d)
