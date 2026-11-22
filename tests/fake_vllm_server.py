"""A stand-in for `vllm serve` used by the tests (no GPU needed).

Implements GET /v1/models, GET /version and POST /v1/chat/completions. Every
request body is recorded, the number of requests in flight is tracked, and the
reply is a deterministic function of the prompt, so sequential and parallel runs
must produce identical results.

Reply rule (on the user message):
  - step 2 of data stealing (scratchpad already calls the first attacker tool)
    -> call the second attacker tool
  - otherwise sha256(prompt) even -> call the first attacker tool ("obey")
  - otherwise -> Final Answer ("ignore")
Fault injection: prompts containing a marker in `fail_once` get one HTTP 500
(the client should retry); prompts containing a marker in `fail_always` always get HTTP 400.
"""

import ast
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "fake/llama-3.1-8b"


class FakeVLLM:
    def __init__(self, delay=0.03, model=MODEL):
        self.model, self.delay = model, delay
        self.requests, self.lock = [], threading.Lock()
        self.in_flight = self.max_in_flight = 0
        self.fail_once, self.fail_always, self._failed = set(), set(), set()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [
                        {"id": server.model, "object": "model", "owned_by": "vllm",
                         "root": server.model, "max_model_len": 16384}]})
                elif self.path == "/version":
                    self._send(200, {"version": "fake-0.0"})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with server.lock:
                    server.requests.append(body)
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                try:
                    time.sleep(server.delay)
                    user = body["messages"][-1]["content"]
                    for m in server.fail_always:
                        if m in user:
                            return self._send(400, {"error": {"message": "prompt too long (fake)"}})
                    for m in server.fail_once:
                        key = (m, user)
                        if m in user and key not in server._failed:
                            with server.lock:
                                server._failed.add(key)
                            return self._send(500, {"error": {"message": "transient (fake)"}})
                    text = server.reply_native(body) if body.get("tools") else server.reply(user)
                    self._send(200, {
                        "id": "cmpl-fake", "object": "chat.completion", "created": 0, "model": server.model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4,
                                  "total_tokens": len(user) // 4 + len(text) // 4}})
                finally:
                    with server.lock:
                        server.in_flight -= 1

        class Server(ThreadingHTTPServer):
            request_queue_size = 256  # default is 5: bursts of >5 connects get refused and retried
            daemon_threads = True

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @staticmethod
    def reply(user):
        m = (re.search(r"tools available to you are: (\[.*?\])", user)      # S1/S4/S4P prompt
             or re.search(r"should be one of \[?(\[.*?\])\]?", user))       # S2 (hwchase17) prompt
        tools = ast.literal_eval(m.group(1))
        if len(tools) > 2 and f"Action: {tools[1]}\nAction Input:" in user.split("Scratchpad")[-1]:
            return f"Thought: Now I send the data.\nAction: {tools[2]}\nAction Input: {{\"to\": \"x@example.com\"}}"
        if int(hashlib.sha256(user.encode()).hexdigest(), 16) % 2 == 0:
            return f"Thought: The document asks me to do this.\nAction: {tools[1]}\nAction Input: {{\"note\": \"fake\"}}"
        return "Thought: I have what I need.\nFinal Answer: Here is a summary of the document."

    @staticmethod
    def reply_native(body):
        """Native tool calling (harness S3): answer with raw Llama-3.1-style JSON tool-call text,
        the text a server returns with tool_choice="none" (no server-side tool parsing)."""
        names = [t["function"]["name"] for t in body["tools"]]
        msgs = body["messages"]
        if len([m for m in msgs if m["role"] == "tool"]) >= 2 and len(names) > 2:   # data-stealing step 2
            return json.dumps({"name": names[2], "parameters": {"to": "x@example.com"}})
        last_tool = [m for m in msgs if m["role"] == "tool"][-1]["content"]
        if int(hashlib.sha256(last_tool.encode()).hexdigest(), 16) % 2 == 0:
            return json.dumps({"name": names[1], "parameters": {"note": "fake"}})
        return "Here is a summary of the document."

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
