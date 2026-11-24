"""Loopback web chat for a single Qwen2.5-VL SGLang engine."""
import asyncio
import base64
import hmac
import json
import os
import re
import sqlite3
import tempfile
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from time import perf_counter, time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL.Image import DecompressionBombError

from api.client import user_message
from deployment.prefill_queue_features import predict
from preprocessing.image_budget import prepare_image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "frontend/chat"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TURNS = 10
MAX_NEW_TOKENS = 384
SESSION_COOKIE = "vision_session"
RULES = json.loads((ROOT / "configs/chat-scheduling.json").read_text())
auth_file = os.getenv("CHAT_AUTH_FILE")
AUTHORIZATION = (
    b"Basic " + base64.b64encode(Path(auth_file).read_bytes().strip())
    if auth_file else None
)


@asynccontextmanager
async def lifespan(app):
    from transformers import AutoTokenizer

    app.state.tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], local_files_only=True)
    app.state.active = set()
    app.state.db = Path(os.environ.get("CHAT_DB", ROOT / "results/chat-engine/chat.sqlite3"))
    app.state.db.parent.mkdir(parents=True, exist_ok=True)
    with database(app) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
            updated REAL NOT NULL, prefix INTEGER NOT NULL DEFAULT 0,
            history TEXT NOT NULL DEFAULT '[]')""")
        db.execute("CREATE INDEX IF NOT EXISTS owner_updated ON chats(owner, updated DESC)")
    async with httpx.AsyncClient(timeout=180) as http:
        app.state.http = http
        yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")


@contextmanager
def database(app):
    db = sqlite3.connect(app.state.db)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


@app.middleware("http")
async def browser_session(request, call_next):
    if AUTHORIZATION and not hmac.compare_digest(
        request.headers.get("authorization", "").encode(), AUTHORIZATION
    ):
        return JSONResponse(
            {"detail": "请输入演示账号和密码"}, status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Visual Chat", charset="UTF-8"',
                     "Cache-Control": "no-store"},
        )
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD") and origin and urlsplit(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "请求来源与当前站点不一致"}, status_code=403)
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdecimal() and int(content_length) > 4 * MAX_IMAGE_BYTES + 65536:
        return JSONResponse({"detail": "附件总大小超出限制"}, status_code=413)
    owner = request.cookies.get(SESSION_COOKIE, "")
    fresh = not re.fullmatch(r"[0-9a-f]{32}", owner)
    request.state.owner = uuid4().hex if fresh else owner
    response = await call_next(request)
    if fresh and request.url.path != "/health":
        response.set_cookie(SESSION_COOKIE, request.state.owner, max_age=60 * 60 * 24 * 30,
                            httponly=True, samesite="strict", secure=request.url.scheme == "https")
    if request.url.path.startswith("/api"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/")
def index():
    return FileResponse(ASSETS / "index.html")


@app.get("/health")
async def health():
    base = os.getenv("VLM_BASE_URL", "http://127.0.0.1:30002/v1")
    try:
        result = await app.state.http.get(base.rstrip("/").removesuffix("/v1") + "/health", timeout=3)
        result.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, "模型服务未就绪") from exc
    return {"status": "ready", "policy": "uncached_tokens_aging", "aging_ms": RULES["overdue_ms"]}


@app.get("/api/chats")
def chats(request: Request):
    with database(app) as db:
        rows = db.execute("SELECT id,title FROM chats WHERE owner=? ORDER BY updated DESC",
                          (request.state.owner,)).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/chats", status_code=201)
def create_chat(request: Request):
    chat = uuid4().hex
    with database(app) as db:
        db.execute("INSERT INTO chats(id,owner,title,updated) VALUES (?,?,?,?)",
                   (chat, request.state.owner, "新对话", time()))
    return {"id": chat, "title": "新对话"}


def get_chat(chat, owner):
    with database(app) as db:
        row = db.execute("SELECT * FROM chats WHERE id=? AND owner=?", (chat, owner)).fetchone()
    if row is None:
        raise HTTPException(404, "会话不存在")
    return row


@app.get("/api/chats/{chat}")
def conversation(chat: str, request: Request):
    row = get_chat(chat, request.state.owner)
    return {"id": chat, "title": row["title"], "history": json.loads(row["history"])}


@app.delete("/api/chats/{chat}", status_code=204)
async def delete_chat(chat: str, request: Request):
    get_chat(chat, request.state.owner)
    if chat in app.state.active:
        raise HTTPException(409, "会话正在生成回答")
    with database(app) as db:
        db.execute("DELETE FROM chats WHERE id=? AND owner=?", (chat, request.state.owner))


async def prepare_upload(upload, question):
    raw = await upload.read(MAX_IMAGE_BYTES + 1)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("每张图片须小于 8 MiB")
    with tempfile.NamedTemporaryFile() as file:
        file.write(raw)
        file.flush()
        return await asyncio.to_thread(prepare_image, file.name, "medium", question)


def estimate_total(messages, visual_tokens, tokenizer):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(rendered, add_special_tokens=False)) - rendered.count("<|image_pad|>") + visual_tokens


def stream_event(ndjson, kind, **fields):
    line = json.dumps({"type": kind, **fields}, ensure_ascii=False)
    return line + "\n" if ndjson else "data:" + line + "\n\n"


async def generate(request, chat, row, history, question, uploads, ndjson):
    def event(kind, **fields):
        return stream_event(ndjson, kind, **fields)

    started = perf_counter()
    first = None
    text = ""
    usage = {}
    reason = None
    try:
        yield event("status", message="正在处理请求")
        images = [await prepare_upload(upload, question) for upload in uploads]
        message = user_message(question, images)
        visual = sum(turn.get("visual_tokens", 0) for turn in history)
        new_visual = sum(max(1, round(im.metadata["width"] / 28)) * max(1, round(im.metadata["height"] / 28))
                         for im in images)
        messages = [{"role": turn["role"], "content": turn["content"]} for turn in history] + [message]
        total = await asyncio.to_thread(estimate_total, messages, visual + new_visual, app.state.tokenizer)
        if total + MAX_NEW_TOKENS > 7900:
            raise ValueError("图片和历史已接近上下文上限，请新建对话或减少附件")
        uncached = max(1, total - row["prefix"])
        cost = predict({"uncached": uncached}, RULES["token_fit"], floor=None)
        rid = f"stage2-mm-session-sched:tokens:{round(cost * 1000)}:{uuid4().hex}"
        yield event("queued", uncached_tokens_est=uncached, policy="uncached_tokens_aging")
        payload = {
            "model": os.getenv("VLM_MODEL", "qwen25vl-7b"), "messages": messages,
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0, "rid": rid, "cache_salt": chat,
        }
        base = os.getenv("VLM_BASE_URL", "http://127.0.0.1:30002/v1").rstrip("/")
        complete = False
        async with app.state.http.stream("POST", f"{base}/chat/completions", json=payload) as upstream:
            upstream.raise_for_status()
            async for line in upstream.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    complete = True
                    break
                chunk = json.loads(raw)
                if chunk.get("error"):
                    raise RuntimeError("模型服务返回生成错误")
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    reason = choice.get("finish_reason") or reason
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        if first is None:
                            first = perf_counter()
                        text += delta
                        yield event("delta", text=delta)
        if not complete or first is None or not reason:
            raise RuntimeError("模型返回不完整，请重试")
        stats = {"ttft_ms": round((first - started) * 1000), "uncached_tokens_est": uncached,
                 "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                 "finish_reason": reason, "request_id": rid}
        history.extend([{**message, "visual_tokens": new_visual},
                        {"role": "assistant", "content": text, "stats": stats}])
        title = row["title"] if row["prefix"] else question[:32]
        with database(app) as db:
            db.execute("UPDATE chats SET title=?,updated=?,prefix=?,history=? WHERE id=? AND owner=?",
                       (title, time(), total, json.dumps(history, ensure_ascii=False), chat, request.state.owner))
        yield event("done", stats=stats, title=title)
    except (ValueError, OSError, RuntimeError, httpx.HTTPError, DecompressionBombError) as exc:
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "图片无法读取或模型暂时不可用，请重试"
        yield event("error", message=message)
    finally:
        app.state.active.discard(chat)


@app.post("/api/chats/{chat}/stream")
async def stream(chat: str, request: Request, question: str = Form(...), images: list[UploadFile] = File(default=[])):
    row = get_chat(chat, request.state.owner)
    question = question.strip()
    if not question or len(question) > 4000 or len(images) > 4:
        raise HTTPException(422, "问题需为 1–4000 字符，每轮最多 4 张图片")
    history = json.loads(row["history"])
    if len(history) >= MAX_TURNS * 2:
        raise HTTPException(409, "已达到 10 轮，请开始新对话")
    if chat in app.state.active or len(app.state.active) >= 24:
        raise HTTPException(409, "会话正在生成回答或系统繁忙")
    app.state.active.add(chat)
    ndjson = "application/x-ndjson" in request.headers.get("accept", "")
    return StreamingResponse(generate(request, chat, row, history, question, images, ndjson),
                             media_type="application/x-ndjson" if ndjson else "text/event-stream", headers={
                                 "Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no", "Vary": "Accept",
                             })
