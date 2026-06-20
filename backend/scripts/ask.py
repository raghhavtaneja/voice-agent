"""Quick manual tester for the webhook: text in -> grounded answer out.

Prints the RAG chunks (with scores + KEEP/DROP against the relevance floor)
before the answer, so you can see exactly what context the LLM did/didn't get.

Usage (from backend/):
    .venv/bin/python -m scripts.ask what is oci
    .venv/bin/python -m scripts.ask --chunks what is oci   # retrieval only, no LLM call
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from uuid import UUID

# Make `config`/`services` importable no matter how the script is launched
# (python scripts/ask.py, python -m scripts.ask, or from any directory).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from services.rag import _index, _namespace  # noqa: E402

AGENT_ID = "c1748858-632f-4977-a3db-02f8c82f19ae"  # OCI agent
URL = f"http://localhost:8000/webhook/vapi/{AGENT_ID}/chat/completions"

# `-c` / `--chunks` shows only the retrieval (no LLM call) so you can debug
# chunks/scores without spending the tiny Gemini free-tier daily quota.
args = sys.argv[1:]
chunks_only = bool(args) and args[0] in ("-c", "--chunks")
if chunks_only:
    args = args[1:]
question = " ".join(args) or "What is OCI?"


def show_chunks(query: str) -> None:
    """Raw top_k retrieval with scores, marking what the floor keeps vs drops."""
    floor = settings.min_relevance_score
    print(f"\n=== RAG retrieval  (top_k={settings.top_k}, min_score={floor}) ===")
    res = _index().search(
        namespace=_namespace(UUID(AGENT_ID)),
        top_k=settings.top_k,
        inputs={"text": query},
    )
    for hit in res.result.hits:
        mark = "KEEP" if hit.score >= floor else "DROP"
        text = re.sub(r"\s+", " ", hit.fields["text"]).strip()
        page = hit.fields.get("page_number")
        print(f"  [{mark}] score={hit.score:.3f} p{page}: {text[:160]}")
    print("=" * 60)


def ask(query: str) -> None:
    body = json.dumps(
        {"model": "x", "stream": False, "messages": [{"role": "user", "content": query}]}
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        print("\nANSWER:", data["choices"][0]["message"]["content"])
        usage = data.get("usage")
        if usage:
            print(
                f"TOKENS: prompt={usage['prompt_tokens']} "
                f"output={usage['completion_tokens']} total={usage['total_tokens']}"
            )
    except urllib.error.HTTPError as e:
        print(f"\nHTTP {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        print(f"\nCould not reach server (is it running on :8000?): {e.reason}")


print("Q:", question)
show_chunks(question)
if chunks_only:
    print("\n(--chunks: skipped LLM call to save Gemini quota)")
else:
    ask(question)
