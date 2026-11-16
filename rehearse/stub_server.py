"""stub_server.py PORT -- rehearsal stand-in for `vllm serve` (no GPU): /health, /start_profile, /stop_profile, and any
OpenAI-style POST answered with a tiny completion (SSE when "stream" is set). Prints the log lines run_all greps for."""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

print("INFO Model loading took 1.00 GiB memory and 1.0 seconds", flush=True)
print("INFO GPU KV cache size: 1,000 tokens, Maximum concurrency for 262,144 tokens per request: 1.00x", flush=True)
print("INFO SpecDecoding metrics: Mean acceptance length: 1.00, Accepted: 0 tokens, Drafted: 7 tokens", flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._json({"object": "list", "data": [{"id": "inkling"}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path in ("/start_profile", "/stop_profile"):
            return self._json({})
        k = int(body.get("max_tokens") or 8)
        usage = {"prompt_tokens": 1, "completion_tokens": k, "total_tokens": k + 1}
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for i in range(min(k, 8)):
                ch = {"choices": [{"index": 0, "delta": {"content": "ok "}, "text": "ok ", "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode()); self.wfile.flush(); time.sleep(0.01)
            fin = {"choices": [{"index": 0, "delta": {}, "text": "", "finish_reason": "length"}], "usage": usage}
            self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode()); return
        self._json({"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "text": "ok",
                                 "logprobs": None, "finish_reason": "length"}], "usage": usage})


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
