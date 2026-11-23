"""OpenAI-compatible SSE client; timings are client-observed, not GPU kernel times."""
import json
import os
import time
from dataclasses import dataclass, asdict
import httpx

@dataclass
class Completion:
    text: str
    ttft_s: float
    e2e_s: float
    tpot_s: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    content_chunks: int
    finish_reason: str | None
    def to_dict(self):
        return asdict(self)

class VLMClient:
    def __init__(self, base_url=None, model=None, timeout=180, transport=None):
        self.base_url = (base_url or os.getenv("VLM_BASE_URL", "http://127.0.0.1:30000/v1")).rstrip("/")
        self.model = model or os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.http = httpx.Client(timeout=timeout, transport=transport,
            headers={"Authorization":"Bearer "+os.getenv("VLM_API_KEY","EMPTY")})
    def __enter__(self): return self
    def __exit__(self,*args): self.http.close()
    def flush_cache(self):
        url=self.base_url.removesuffix("/v1")+"/flush_cache"
        self.http.post(url).raise_for_status()
    def complete(self, messages, max_tokens=128):
        if not messages or max_tokens < 1:
            raise ValueError("Messages and positive max_tokens required")
        start=time.perf_counter(); first=None; last=None; text=""; usage={}; chunks=0; reason=None; done=False
        with self.http.stream("POST",self.base_url+"/chat/completions",json={
            "model":self.model,"messages":messages,"temperature":0,
            "max_tokens":max_tokens,"stream":True,"stream_options":{"include_usage":True}}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"): continue
                raw=line[5:].strip()
                if raw=="[DONE]": done=True; break
                event=json.loads(raw)
                if "error" in event: raise RuntimeError(str(event["error"]))
                usage=event.get("usage") or usage
                for choice in event.get("choices",[]):
                    reason=choice.get("finish_reason") or reason
                    delta=choice.get("delta",{}).get("content") or ""
                    if delta:
                        last=time.perf_counter()
                        if first is None: first=last
                        chunks+=1; text+=delta
        end=time.perf_counter()
        if not done or first is None or reason is None:
            raise RuntimeError("Incomplete or empty SSE response")
        tokens=usage.get("completion_tokens")
        # Chunks may contain multiple tokens. Never use chunk count as token count.
        tpot=(last-first)/(tokens-1) if tokens and tokens>1 and last>first else None
        return Completion(text,first-start,end-start,tpot,tokens,usage.get("prompt_tokens"),chunks,reason)

def user_message(question, images):
    if not question or not question.strip(): raise ValueError("问题不能为空")
    if len(images)>8: raise ValueError("最多支持 8 张图片")
    return {"role":"user","content":[{"type":"image_url","image_url":{"url":im.url}} for im in images]
        +[{"type":"text","text":question.strip()}]}
