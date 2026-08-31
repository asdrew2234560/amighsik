"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.
"""

from __future__ import annotations

import threading
import time
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware  # <-- اضافه شد
from .config import DEFAULT_MODEL
from .schemas import ChatCompletionRequest
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient

from .config import (
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    resolve_model_type,
)
from .openai_format import completion_response, messages_to_prompt, stream_chunks
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest

load_dotenv()


app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.1.0")

# ===================== CORS MIDDLEWARE =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://m.mruclml.ir",
        "https://mruclml.ir",
        "https://round-dust-7dbf.abdolghasems.workers.dev",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# ===========================================================

install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.

    This touches Playwright's sync API, so callers must invoke it OFF the event
    loop (via run_in_threadpool); calling it inside the asyncio loop raises
    "Playwright Sync API inside the asyncio loop"."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
            for name in MODEL_MAP
        ],
    }


@app.post("/v1/responses")
async def responses_proxy(request: Request):
    """
    Compatibility shim for clients calling the newer /v1/responses endpoint.
    Accepts either `messages` or `input` and forwards to the existing chat_completions handler.
    """
    try:
        body = await request.json()
    except Exception:
        return _error("Request body must be JSON", status=400, err_type="invalid_request_error")

    model = body.get("model", DEFAULT_MODEL)
    stream = bool(body.get("stream", False))
    conversation_id = body.get("conversation_id")
    thinking = bool(body.get("thinking", False))
    search = bool(body.get("search", False))

    # Accept either messages (chat API) or input (Responses API)
    messages = None
    if body.get("messages"):
        messages = body["messages"]
    elif "input" in body and body["input"] is not None:
        inp = body["input"]
        if isinstance(inp, list):
            content = "\n".join(str(x) for x in inp)
        else:
            content = str(inp)
        messages = [{"role": "user", "content": content}]
    else:
        return _error("`messages` or `input` must be provided", status=400, err_type="invalid_request_error")

    try:
        chat_req = ChatCompletionRequest(
            model=model,
            messages=messages,
            stream=stream,
            conversation_id=conversation_id,
            thinking=thinking,
            search=search,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_tokens"),
            user=body.get("user"),
        )
    except Exception as e:
        return _error(f"Invalid request: {e}", status=400, err_type="invalid_request_error")

    try:
        return await chat_completions(chat_req)
    except Exception as e:
        # Print traceback to server logs for debugging (stdout/stderr)
        import traceback
        traceback.print_exc()
        return _error(f"Responses proxy failed: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    prompt = messages_to_prompt(req.messages)

    try:
        # Off the event loop: get_client() uses Playwright's sync API, which
        # errors if run inside the asyncio loop.
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    if req.stream:
        def gen():
            stream = client.stream(
                prompt, conversation_id=req.conversation_id,
                model=model_type, thinking=req.thinking, search=req.search,
            )
            yield from stream_chunks(req.model, stream)

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat, prompt, req.conversation_id,
            model_type, req.thinking, req.search,
        )
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    return completion_response(req.model, reply.text, prompt, reply.conversation_id)
