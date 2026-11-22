"""Tests for src/evaluate_harness.py (scaffolds S1-S4P) against the fake vLLM server.
Run from the repo root:  PYTHONPATH=. pytest -q tests"""

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))  # evaluate_harness.py imports its siblings by bare name

import evaluate_harness as hrun  # noqa: E402
from harnesses import HARNESSES, SPOTLIGHT_NOTICE  # noqa: E402
from params import parse_arguments  # noqa: E402
from tests.fake_vllm_server import FakeVLLM, MODEL  # noqa: E402

CELL = "data/channels/pdf/pymupdf/white"


def _params(url, harness, **kw):
    p = {"model_type": "VLLM", "model_name": MODEL, "setting": "base", "harness": harness,
         "only_first_step": True, "use_cache": False, "only_get_score": False, "data_dir": CELL,
         "num_workers": 4, "vllm_base_url": url, "max_tokens": 512, "temperature": 0.0, "limit": 4}
    p.update(kw)
    return p


def _out_dir(p):
    return (f"./results/harness_VLLM_{p['model_name']}_{p['harness']}_channels-pdf-pymupdf-white"
            + (f"_limit{p['limit']}" if p.get("limit") else ""))


def _rows(p, attack):
    return [json.loads(l) for l in open(os.path.join(_out_dir(p), f"test_cases_{attack}_base.json"))]


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    shutil.rmtree(os.path.join(ROOT, "results", "harness_VLLM_fake"), ignore_errors=True)
    yield
    shutil.rmtree(os.path.join(ROOT, "results", "harness_VLLM_fake"), ignore_errors=True)


@pytest.fixture
def server():
    with FakeVLLM() as s:
        yield s


def test_cli_accepts_harness_options(monkeypatch):
    """Regression: parse_arguments(agent_type="harness") defined no options at all."""
    monkeypatch.setattr(sys, "argv", ["evaluate_harness.py", "--model_type", "VLLM", "--model_name", MODEL,
                                      "--setting", "base", "--harness", "S3", "--only_first_step",
                                      "--num_workers", "16", "--limit", "5"])
    p = parse_arguments(agent_type="harness")
    assert p["harness"] == "S3" and p["num_workers"] == 16 and p["limit"] == 5 and "prompt_type" not in p


def test_cli_end_to_end(server):
    """The documented command line runs (subprocess, exactly as on the H200)."""
    env = dict(os.environ, PYTHONPATH=".")
    env.pop("OPENAI_API_KEY", None)
    r = subprocess.run([sys.executable, "src/evaluate_harness.py", "--model_type", "VLLM", "--model_name", MODEL,
                        "--setting", "base", "--harness", "S3", "--only_first_step", "--num_workers", "4",
                        "--limit", "2", "--data_dir", CELL, "--vllm_base_url", server.url],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[dh] 2/2 written, 0 failed" in r.stdout and "[ds] 2/2 written, 0 failed" in r.stdout


@pytest.mark.parametrize("harness", ["S1", "S2", "S3", "S4", "S4P"])
def test_every_harness_runs_on_vllm(server, harness):
    p = _params(server.url, harness)
    hrun.main(p)
    m = json.load(open(os.path.join(_out_dir(p), "run_manifest.json")))
    assert m["files"]["dh"]["n_failed"] == 0 and m["files"]["ds"]["n_failed"] == 0
    rows = _rows(p, "dh") + _rows(p, "ds")
    assert len(rows) == 8 and all(r["eval"] in ("succ", "unsucc") for r in rows)
    assert {"succ", "unsucc"} <= {r["eval"] for r in rows}


def test_react_harnesses_send_no_tools(server):
    hrun.main(_params(server.url, "S4"))
    for r in server.requests:
        assert "tools" not in r and "tool_choice" not in r
        assert [m["role"] for m in r["messages"]] == ["system", "user"]
        assert SPOTLIGHT_NOTICE in r["messages"][0]["content"]
        assert "<tool_output>" in r["messages"][1]["content"]


def test_s3_sends_schemas_with_tool_choice_none_and_string_arguments(server):
    p = _params(server.url, "S3")
    hrun.main(p)
    for r in server.requests:
        assert r["tool_choice"] == "none" and len(r["tools"]) >= 2
        assert all(t["type"] == "function" for t in r["tools"])
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant", "tool"]
        call = r["messages"][2]["tool_calls"][0]["function"]
        assert isinstance(call["arguments"], str) and isinstance(json.loads(call["arguments"]), dict)
    # The harness's own messages are not modified (copy made for the wire format).
    item = json.load(open(os.path.join(CELL, "test_cases_dh_base.json")))[0]
    from utils import get_tool_dict
    msgs, _ = HARNESSES["S3"].build(item, get_tool_dict(), get_tool_dict(gpt_format=True))
    assert isinstance(msgs[2]["tool_calls"][0]["function"]["arguments"], dict)


def test_s3_parses_raw_tool_call_text(server):
    p = _params(server.url, "S3", limit=6)
    hrun.main(p)
    rows = _rows(p, "dh")
    succ = [r for r in rows if r["eval"] == "succ"]
    assert succ and all(r["tool_calls"][0]["name"] == r["Attacker Tools"][0] for r in succ)


def test_s3_data_stealing_step2(server, monkeypatch):
    monkeypatch.setattr(hrun, "get_simulated_attacker_tool_response", lambda *a: '{"result": "simulated"}')
    p = _params(server.url, "S3", only_first_step=False, limit=8)
    hrun.main(p)
    s1 = [r for r in _rows(p, "ds") if r["eval"] == "succ"]
    assert s1 and all(r["eval Step 2"] == "succ" for r in s1)
    step2 = [r for r in server.requests if len([m for m in r["messages"] if m["role"] == "tool"]) == 2]
    assert step2 and all(isinstance(m["tool_calls"][0]["function"]["arguments"], str)
                         for r in step2 for m in r["messages"] if m.get("tool_calls"))
