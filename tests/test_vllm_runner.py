"""Tests for the vLLM wrapper and the parallel runner, against a fake vLLM server.
Run from the repo root:  PYTHONPATH=. pytest -q tests"""

import json
import os
import shutil
import threading
import time

import pytest

import src.evaluate_prompted_agent as runner
from src.models import MODELS, VLLMModel
from src.prompts.agent_prompts import PROMPT_DICT
from tests.fake_vllm_server import FakeVLLM, MODEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = "data/channels/pdf/pymupdf/white"  # 30 dh + 32 ds cases


def _params(url, **kw):
    p = {"model_type": "VLLM", "model_name": MODEL, "setting": "base", "prompt_type": "InjecAgent",
         "only_first_step": True, "use_cache": False, "only_get_score": False, "data_dir": CELL,
         "num_workers": 1, "vllm_base_url": url, "max_tokens": 512, "temperature": 0.0}
    p.update(kw)
    return p


def _out_dir(params):
    d = f"./results/prompted_VLLM_{params['model_name']}_InjecAgent_channels-pdf-pymupdf-white"
    return d + (f"_limit{params['limit']}" if params.get("limit") else "")


def _rows(out_dir, attack):
    return [json.loads(l) for l in open(os.path.join(out_dir, f"test_cases_{attack}_base.json"))]


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    shutil.rmtree(os.path.join(ROOT, "results", "prompted_VLLM_fake"), ignore_errors=True)
    yield
    shutil.rmtree(os.path.join(ROOT, "results", "prompted_VLLM_fake"), ignore_errors=True)


@pytest.fixture
def server():
    with FakeVLLM() as s:
        yield s


# ---------- what gets sent to the server ----------

def test_requests_mirror_hfchat_and_pin_sampling(server):
    runner.main(_params(server.url, limit=3))
    assert len(server.requests) == 6  # 3 dh + 3 ds, step 1 only
    sys_prompt, _ = PROMPT_DICT["InjecAgent"]
    for r in server.requests:
        assert r["model"] == MODEL
        assert [m["role"] for m in r["messages"]] == ["system", "user"]
        assert r["messages"][0]["content"] == sys_prompt
        assert r["temperature"] == 0.0 and r["top_p"] == 1.0 and r["max_tokens"] == 512
        assert r["top_k"] == -1 and r["min_p"] == 0.0 and r["repetition_penalty"] == 1.0
        assert "stop" not in r and "seed" not in r and "chat_template_kwargs" not in r


def test_optional_request_fields(server):
    runner.main(_params(server.url, limit=1, seed=7, stop_at_observation=True,
                        chat_template_kwargs='{"enable_thinking": false}', model_name=MODEL))
    r = server.requests[0]
    assert r["seed"] == 7 and r["stop"] == ["\nObservation:"]
    assert r["chat_template_kwargs"] == {"enable_thinking": False}


def test_user_prompt_is_identical_to_original_runner(server):
    """The vLLM path must send exactly the prompt the original runner builds."""
    runner.main(_params(server.url, limit=2))
    sys_p, user_p = PROMPT_DICT["InjecAgent"]
    from src.utils import get_tool_dict
    tool_dict = get_tool_dict()
    item = json.load(open(os.path.join(CELL, "test_cases_dh_base.json")))[0]

    class Capture:
        def prepare_input(self, s, u): return (s, u)
        def call_model(self, x): return x
    s, u = runner.predict_one_case(sys_p, user_p, {}, item, Capture(), tool_dict)
    assert server.requests[0]["messages"] == [{"role": "system", "content": s}, {"role": "user", "content": u}]


# ---------- parallel runs ----------

def test_parallel_matches_sequential_and_keeps_order(server):
    seq = _params(server.url, model_name=MODEL)
    runner.main(seq)
    seq_rows = {a: _rows(_out_dir(seq), a) for a in ("dh", "ds")}
    assert server.max_in_flight == 1

    server.max_in_flight = 0
    par = dict(seq, num_workers=16)
    shutil.rmtree(_out_dir(seq))
    runner.main(par)
    par_rows = {a: _rows(_out_dir(par), a) for a in ("dh", "ds")}
    assert server.max_in_flight > 1, "requests were not concurrent"
    for a in ("dh", "ds"):
        src = json.load(open(os.path.join(CELL, f"test_cases_{a}_base.json")))
        assert [r["Case ID"] for r in par_rows[a]] == [c["Case ID"] for c in src]
        assert par_rows[a] == seq_rows[a]
    evals = [r["eval"] for a in ("dh", "ds") for r in par_rows[a]]
    assert {"succ", "unsucc"} <= set(evals) and "invalid" not in evals


def test_parallel_is_faster(server):
    server.delay = 0.05
    t = time.time(); runner.main(_params(server.url, limit=20)); seq = time.time() - t
    shutil.rmtree(_out_dir(_params(server.url, limit=20)))
    t = time.time(); runner.main(_params(server.url, limit=20, num_workers=20)); par = time.time() - t
    assert par < seq / 3


def test_call_info_and_manifest(server):
    p = _params(server.url, limit=2)
    runner.main(p)
    row = _rows(_out_dir(p), "dh")[0]
    assert row["call_info"]["finish_reason"] == "stop" and row["call_info"]["prompt_tokens"] > 0
    m = json.load(open(os.path.join(_out_dir(p), "run_manifest.json")))
    assert m["server_info"]["vllm_version"] == "fake-0.0" and m["server_info"]["max_model_len"] == 16384
    assert m["request_args"]["temperature"] == 0.0 and m["extra_body"]["repetition_penalty"] == 1.0
    assert m["files"]["dh"] == {"test_case_file": os.path.join(CELL, "test_cases_dh_base.json"), "n_cases": 2,
                                "n_written": 2, "n_failed": 0, "n_from_cache": 0, "elapsed_s": m["files"]["dh"]["elapsed_s"]}
    assert m["scores"]["#Test Case"] == 4


# ---------- failures, retries, cache ----------

def test_transient_server_error_is_retried(server):
    server.fail_once.add("summar")  # every prompt asks for a summary -> each fails once
    p = _params(server.url, limit=3, num_workers=3)
    runner.main(p)
    assert len(_rows(_out_dir(p), "dh")) == 3 and len(server.requests) == 12  # 6 cases x (fail + retry)


def test_permanent_failure_is_counted_not_hidden(server, capsys):
    # The user instruction of dh case 0 is unique among the first 3 dh + 3 ds cases (different documents).
    first = json.load(open(os.path.join(CELL, "test_cases_dh_base.json")))[0]["User Instruction"]
    server.fail_always.add(first)
    p = _params(server.url, limit=3, num_workers=3)
    runner.main(p)
    m = json.load(open(os.path.join(_out_dir(p), "run_manifest.json")))
    assert m["files"]["dh"]["n_failed"] == 1 and m["files"]["dh"]["n_written"] == 2
    assert "WARNING: 1 case(s) failed" in capsys.readouterr().out


def test_use_cache_resumes_without_new_requests(server):
    p = _params(server.url, limit=4, num_workers=4)
    runner.main(p)
    first = {a: _rows(_out_dir(p), a) for a in ("dh", "ds")}
    n = len(server.requests)
    runner.main(dict(p, use_cache=True))
    assert len(server.requests) == n
    for a in ("dh", "ds"):
        again = _rows(_out_dir(p), a)
        assert [r["output"] for r in again] == [r["output"] for r in first[a]]
        assert [r["eval"] for r in again] == [r["eval"] for r in first[a]]


def test_limit_writes_to_separate_folder(server):
    p = _params(server.url, limit=2)
    runner.main(p)
    assert _out_dir(p).endswith("_limit2") and os.path.isdir(_out_dir(p))
    assert not os.path.isdir(_out_dir(dict(p, limit=None)))


# ---------- data stealing step 2 ----------

def test_step2_runs_in_parallel_with_simulator_serialized(server, monkeypatch):
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fake_simulator(tool, params, tool_dict):
        with lock:
            active["now"] += 1; active["max"] = max(active["max"], active["now"])
        time.sleep(0.02)
        with lock:
            active["now"] -= 1
        return '{"result": "simulated"}'

    monkeypatch.setattr(runner, "get_simulated_attacker_tool_response", fake_simulator)
    p = _params(server.url, only_first_step=False, num_workers=16)
    runner.main(p)
    ds = _rows(_out_dir(p), "ds")
    s1 = [r for r in ds if r["eval"] == "succ"]
    assert s1 and all(r["eval Step 2"] == "succ" and "call_info Step 2" in r for r in s1)
    assert active["max"] == 1, "simulator ran concurrently"


# ---------- guards ----------

def test_local_gpu_models_refuse_parallel():
    with pytest.raises(ValueError, match="not thread-safe"):
        runner.main({"model_type": "HFChat", "model_name": "x", "setting": "base", "prompt_type": "InjecAgent",
                     "only_first_step": True, "use_cache": False, "only_get_score": False, "num_workers": 4})


def test_wrong_model_name_fails_fast(server):
    with pytest.raises(ValueError, match="Pass the served model name"):
        VLLMModel({"model_name": "meta-llama/Llama-3.1-8B-Instruct", "vllm_base_url": server.url})


def test_models_import_without_together():
    assert "VLLM" in MODELS and MODELS["VLLM"].supports_concurrency and not MODELS["HFChat"].supports_concurrency


def test_summarize_results_runs(server, capsys, monkeypatch):
    monkeypatch.setattr(runner, "get_simulated_attacker_tool_response", lambda *a: '{"result": "simulated"}')
    p = _params(server.url, only_first_step=False, num_workers=8, limit=6)
    runner.main(p)
    capsys.readouterr()
    from src.summarize_results import summarize
    summarize(_out_dir(p))
    out = capsys.readouterr().out
    assert "12 rows" in out and "finish_reason: {'stop': 12}" in out and "eval (step 2)" in out and "scores:" in out


def test_no_openai_key_needed_for_local_runs(server, monkeypatch):
    """src/utils.py creates its OpenAI client lazily, so a vLLM run needs no OPENAI_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import src.utils
    monkeypatch.setattr(src.utils, "_client", None)
    runner.main(_params(server.url, limit=2))
    assert src.utils._client is None


def test_step2_simulator_uses_lazy_client(tmp_path, monkeypatch):
    """Regression: the simulator must call get_openai_client() (it once called a deleted
    module-level `client`, which crashed every data-stealing step 2)."""
    import src.utils as u
    sent = {}

    class _Completions:
        def create(self, **kw):
            sent.update(kw)
            msg = type("M", (), {"content": "```json\n{\"emails\": []}\n```"})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    fake_client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    monkeypatch.setattr(u, "get_openai_client", lambda: fake_client)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "attacker_simulated_responses.json").write_text("{}")
    tool_dict = u.get_tool_dict()           # read from the repo before switching folders
    monkeypatch.chdir(tmp_path)              # the simulator writes its cache under ./data
    out = u.get_simulated_attacker_tool_response("GmailSearchEmails", "{'keywords': ['x']}", tool_dict)
    assert out == '{"emails": []}' and sent["model"] == "gpt-4-0613"
    assert json.loads((tmp_path / "data" / "attacker_simulated_responses.json").read_text())
