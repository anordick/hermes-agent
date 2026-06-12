#!/usr/bin/env python3
"""
Crucible Lite — Fast Text Parsing Layer

Lets profiles delegate text parsing, extraction, and summarization to a local
Qwen model on LM Studio for fast inline processing.

Architecture:
  Profile (reasoning)
    → runs terminal commands, reads files, captures raw output
    → calls crucible_lite(task, context) with the raw output
    → Qwen parses, extracts, summarizes into clean structured data
    → Profile reasons on the clean data

Qwen does NOT execute tools directly. It receives raw text (terminal output,
file contents, log dumps) via the `context` parameter and returns structured
JSON. This avoids tool-calling loops and keeps Qwen's job focused and fast.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def _log_crucible_call(task: str, context_chars: int, latency_s: float,
                       success: bool, output_chars: int, note: str = "") -> None:
    """Append a markdown row to ~/.farpoint/crucible-log.md with metrics."""
    # Token estimates: ~4 chars/token for text, ~3 for JSON
    tok_in = context_chars // 4
    tok_out = output_chars // 3
    # Reasoning tokens saved: without Crucible DS spends ~50% of input tokens
    # on reasoning. With Crucible, DS spends ~10%. Net savings estimate.
    # Positive = saved tokens, negative = extra cost from JSON expansion
    tok_saved_est = int((tok_in * 0.5) - tok_out)
    
    # Extract profile name from HERMES_HOME (e.g. ~/.hermes/profiles/anvil → anvil)
    _hermes_home = os.environ.get("HERMES_HOME", "")
    if "/profiles/" in _hermes_home:
        _profile = _hermes_home.rstrip("/").split("/")[-1]
    else:
        _profile = "default"
    
    logger.info(
        "crucible task='%.80s' ctx=%d lat=%.2fs ok=%s out=%d tokin=%d tokout=%d "
        "saved=%d profile=%s %s",
        task, context_chars, latency_s, success, output_chars,
        tok_in, tok_out, tok_saved_est, _profile, note,
    )
    try:
        log_path = os.path.expanduser("~/.farpoint/crucible-log.md")
        date_str = time.strftime("%Y-%m-%d")
        short_task = task.replace("\n", " ").strip()[:80]
        verdict = "PASS" if success else "FAIL"
        row = (
            f"| {_profile} | {date_str} | {short_task} | "
            f"{context_chars} | {latency_s:.1f}s | {output_chars} "
            f"| tok_in={tok_in} tok_out={tok_out} saved_est={tok_saved_est} "
            f"| {verdict} | | qwen3.5-2b local |\n"
        )
        with open(log_path, "a") as f:
            f.write(row)
    except Exception as e:
        logger.debug("crucible log append failed: %s", e)

# ---------------------------------------------------------------------------
# LM Studio endpoint
# ---------------------------------------------------------------------------
LM_STUDIO_BASE = os.environ.get(
    "LM_STUDIO_BASE_URL", "http://192.168.0.96:1234"
)
QWEN_MODEL = "qwen/qwen3.5-2b"
QWEN_ENDPOINT = f"{LM_STUDIO_BASE}/v1/chat/completions"

# ---------------------------------------------------------------------------
# Qwen's system prompt — strict parsing/summarization, no reasoning
# ---------------------------------------------------------------------------
QWEN_SYSTEM_PROMPT = (
    "You are the Local Operational Layer \u2014 a text parser and summarizer.\n"
    "You receive raw text (terminal output, file contents, log dumps) and\n"
    "return clean, structured data.\n"
    "\n"
    "Your job:\n"
    "1. Parse messy text into structured JSON data\n"
    "2. Extract key values (numbers, paths, statuses, errors, counts)\n"
    "3. Summarize and organize the data logically\n"
    "4. Return ONLY the data package as valid JSON\n"
    "\n"
    "Rules:\n"
    "- Base your output ONLY on the text provided in context\n"
    "- Do not guess or fabricate data not present in the input\n"
    "- No analysis, no recommendations, no commentary\n"
    "- No decisions about what the data means\n"
    "- No speculation about causes or trends\n"
    "- Return ONLY valid JSON — no markdown, no code fences, no explanations\n"
    "- The calling layer handles all reasoning"
)

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
def _lm_studio_chat(
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> dict:
    """Send a chat request to LM Studio and return the parsed response."""
    import requests as http

    resp = http.post(
        QWEN_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json={
            "model": QWEN_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tool handler — single-turn text parsing
# ---------------------------------------------------------------------------
def crucible_lite(task: str, context: str = "") -> str:
    """
    Send raw text to Qwen for parsing, extraction, and summarization.

    Deepseek runs the actual terminal commands and file reads, then passes
    the raw output to Qwen via the `context` parameter. Qwen returns clean,
    structured JSON data for deepseek to reason on.

    Args:
        task:    What to extract or analyze from the context. Examples:
                 - "Extract disk usage: filesystem, size, used, avail, use%"
                 - "Count errors by type from these log lines"
                 - "Parse this config and return the model section with provider details"
        context: The raw text to parse — terminal output, file contents,
                 log dumps, etc. This is where the data lives.

    Returns:
        JSON string with the clean data package from Qwen.
        The 'qwen_output' field contains the parsed data (dict or string).
    """
    if not task or not task.strip():
        return json.dumps(
            {"error": "task is required", "qwen_output": ""},
            ensure_ascii=False,
        )

    task = task.strip()
    context = context.strip() if context else ""

    if not context:
        return json.dumps(
            {
                "error": "context is required — pass the raw text to parse",
                "qwen_output": "",
            },
            ensure_ascii=False,
        )

    # Build the prompt
    user_content = (
        f"Task: {task}\n\n"
        f"Raw context to parse:\n"
        f"```\n{context}\n```\n\n"
        f"Parse the context above and return ONLY valid JSON "
        f"with the extracted data. No markdown, no explanations."
    )

    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    _t0 = None
    try:
        _t0 = time.monotonic()
        response = _lm_studio_chat(messages)
        _latency = time.monotonic() - _t0
        choices = response.get("choices", [])
        if not choices:
            _log_crucible_call(task, len(context), _latency, False, 0,
                               "no_choices")
            return json.dumps(
                {
                    "error": "LM Studio returned no choices",
                    "qwen_output": "",
                },
                ensure_ascii=False,
            )

        qwen_content = choices[0].get("message", {}).get("content", "")

        # Strip markdown code fences if present
        cleaned = qwen_content.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()

        # Try to parse as JSON
        try:
            parsed = json.loads(cleaned) if cleaned.strip() else {}
            _log_crucible_call(task, len(context), _latency, True,
                               len(cleaned), "json_ok")
            return json.dumps(
                {
                    "qwen_output": parsed,
                    "model": QWEN_MODEL,
                },
                ensure_ascii=False,
            )
        except (json.JSONDecodeError, TypeError):
            _log_crucible_call(task, len(context), _latency, True,
                               len(cleaned), "json_fallback_raw")
            return json.dumps(
                {
                    "qwen_output": cleaned,
                    "model": QWEN_MODEL,
                },
                ensure_ascii=False,
            )

    except Exception as exc:
        _latency = time.monotonic() - _t0 if _t0 is not None else -1
        logger.error("crucible_lite failed: %s", exc)
        _log_crucible_call(task, len(context) if 'context' in dir() else 0,
                           max(_latency, 0), False, 0, f"error={exc}")
        return json.dumps(
            {
                "error": f"Qwen execution failed: {exc}",
                "qwen_output": "",
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Availability check — verify LM Studio is reachable and Qwen is loaded
# ---------------------------------------------------------------------------
def check_qwen_requirements() -> bool:
    """Return True if LM Studio is reachable and has Qwen 3.5-2b loaded."""
    try:
        import requests as http

        resp = http.get(
            f"{LM_STUDIO_BASE}/v1/models",
            timeout=5,
        )
        if resp.status_code != 200:
            return False
        models = resp.json().get("data", [])
        return any(m.get("id") == "qwen/qwen3.5-2b" for m in models)
    except Exception:
        return False


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

QWEN_EXECUTE_SCHEMA = {
    "name": "crucible_lite",
    "description": (
        "Send raw text (terminal output, file contents, log dumps) to the "
        "local Qwen model for parsing, extraction, and summarization. Qwen "
        "returns clean structured JSON data for you to reason on.\n\n"
        "HOW TO USE:\n"
        "1. Run the terminal commands / read the files you need\n"
        "2. Capture the raw output\n"
        "3. Call crucible_lite with a clear task and the raw output as context\n"
        "4. Qwen returns structured JSON — reason on the returned data\n\n"
        "Example:\n"
        "  You run 'df -h' and get raw output. Then call:\n"
        "    crucible_lite(\n"
        "      task='Extract disk usage: filesystem, size, used, avail, use%',\n"
        "      context='<raw df -h output>'\n"
        "    )\n"
        "  Qwen returns: {\"disk_usage\": [{...}, ...]}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "What to extract or analyze from the context text. "
                    "Be specific about the data you need. "
                    "Example: 'Extract disk usage with filesystem, size, "
                    "used, avail, use% as an array of objects'"
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "The raw text to parse — terminal output, file contents, "
                    "log dumps, config files. This is where the data lives. "
                    "Qwen parses this and returns structured JSON."
                ),
            },
        },
        "required": ["task", "context"],
    },
}


# =============================================================================
# Registry
# =============================================================================
from tools.registry import registry, tool_error

registry.register(
    name="crucible_lite",
    toolset="delegation",
    schema=QWEN_EXECUTE_SCHEMA,
    handler=lambda args, **kw: crucible_lite(
        task=args.get("task", ""),
        context=args.get("context", ""),
    ),
    check_fn=check_qwen_requirements,
    emoji="🖥️",
)
