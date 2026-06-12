#!/usr/bin/env python3
"""
|Crucible Deep — Local Tool Execution + Structuring

Lets any profile delegate a goal to Gemma-4-12b-QAT on LM Studio. Gemma:
1. Receives a goal (e.g. "check system health")
2. Runs terminal commands and reads files locally
3. Structures findings into clean JSON
4. Returns the structured data package

DeepSeek never sees raw output — Gemma does the execution and structuring.

Architecture:
  Profile → crucible_deep(goal)
    → Gemma-4-12b-QAT (thinking ON, LM Studio)
    → runs commands, structures findings
    → returns clean JSON
  Profile reasons on the structured data
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LM Studio endpoint
# ---------------------------------------------------------------------------
LM_STUDIO_BASE = os.environ.get(
    "LM_STUDIO_BASE_URL", "http://192.168.0.96:1234"
)
CRUCIBLE_DEEP_MODEL = "google/gemma-4-12b-it-qat"
CRUCIBLE_DEEP_ENDPOINT = f"{LM_STUDIO_BASE}/v1/chat/completions"

# ---------------------------------------------------------------------------
# Tool definitions gemma can use
# ---------------------------------------------------------------------------
_CRUCIBLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command on this Mac system. Returns stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem. Returns content and line count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    }
                },
                "required": ["path"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
def _lm_studio_chat(
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 32768,
    timeout: int = 1200,
) -> dict:
    """Send a chat request to LM Studio with tools and return the response."""
    import requests as http

    resp = http.post(
        CRUCIBLE_DEEP_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json={
            "model": CRUCIBLE_DEEP_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": _CRUCIBLE_TOOLS,
            "tool_choice": "auto",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Instrumentation — log to crucible-log.md
# ---------------------------------------------------------------------------
def _log_call(goal: str, turns: int, latency_s: float,
              success: bool, output_chars: int, note: str = "") -> None:
    """Log a Crucible Deep call to crucible-log.md and agent.log."""
    logger.info(
        "crucible-deep goal='%.60s' turns=%d lat=%.1fs ok=%s out=%d %s",
        goal, turns, latency_s, success, output_chars, note,
    )
    try:
        log_path = os.path.expanduser("~/.farpoint/crucible-log.md")
        date_str = time.strftime("%Y-%m-%d")
        short_goal = goal.replace("\n", " ").strip()[:80]
        verdict = "PASS" if success else "FAIL"
        row = (
            f"| {verdict} | | gemma-4-12b-qat | "
            f"turns={turns} lat={latency_s:.1f}s out={output_chars} "
            f"| {short_goal} |\n"
        )
        with open(log_path, "a") as f:
            f.write(row)
    except Exception as e:
        logger.debug("crucible log append failed: %s", e)


# ---------------------------------------------------------------------------
# Tool handler — run a goal through gemma
def crucible_deep(goal: str) -> str:
    """
    Delegate a goal to Gemma for local execution and structuring.

    Gemma runs terminal commands and reads files on the Mac, then returns
    a structured JSON report with all findings.

    Args:
        goal: What to investigate and report on. Examples:
              - "Check system health: CPU, memory, disk, uptime, services"
              - "Audit Hermes gateway logs for errors in the last hour"
              - "Compare config files between two profiles"

    Returns:
        JSON string with the structured report from Gemma.
        The 'report' field contains the structured findings.
    """
    if not goal or not goal.strip():
        return json.dumps(
            {"error": "goal is required", "report": ""},
            ensure_ascii=False,
        )

    goal = goal.strip()

    # System prompt — gemma decides what to run and when to stop
    system_prompt = (
        "You are the local Crucible operational layer running on the Pi5 Hermes server. "
        "You have been given terminal and read_file tools. "
        "Use these tools to investigate the goal, run the necessary commands "
        "and file reads, then produce a comprehensive structured JSON report "
        "with all findings.\n\n"
        "Rules:\n"
        "1. Run only the commands you need — be efficient\n"
        "2. When you have enough data, stop and produce the report\n"
        "3. Return the report as valid JSON wrapped in ```json...```\n"
        "4. Include all relevant data — don't summarize away details\n"
        "5. Base your report ONLY on actual tool output\n"
        "6. Do not fabricate data you didn't observe"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Goal: {goal}\n\nInvestigate and return a structured JSON report."}
    ]

    _t0 = time.monotonic()
    turn_count = 0
    # POC phase: collect raw tool outputs for DS validation
    _poc_raw_outputs = []

    try:
        for turn_count in range(15):  # Safety limit
            response = _lm_studio_chat(messages)
            choice = response["choices"][0]
            msg = choice["message"]

            if msg.get("tool_calls"):
                # Execute each tool call
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    fn_name = fn["name"]
                    try:
                        fn_args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    # Dispatch the tool — real execution, no simulation
                    import subprocess
                    if fn_name == "terminal":
                        cmd = fn_args.get("command", "")
                        try:
                            tr = subprocess.run(
                                cmd,
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=60,
                            )
                            out = tr.stdout.strip()
                            err = tr.stderr.strip()
                            parts = []
                            if out:
                                parts.append(f"STDOUT:\n{out}")
                            if err:
                                parts.append(f"STDERR:\n{err}")
                            parts.append(f"exit_code={tr.returncode}")
                            tool_output = "\n".join(parts)
                        except subprocess.TimeoutExpired:
                            tool_output = json.dumps({"error": "Command timed out after 60s"})
                        except Exception as e:
                            tool_output = json.dumps({"error": str(e)})
                    elif fn_name == "read_file":
                        path = fn_args.get("path", "")
                        try:
                            with open(path, "r") as f:
                                content = f.read()
                            tool_output = json.dumps({"content": content, "total_lines": content.count("\n") + 1})
                        except Exception as e:
                            tool_output = json.dumps({"error": str(e)})
                    else:
                        tool_output = json.dumps({"error": f"Unknown tool: {fn_name}"})

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": fn_name, "arguments": fn["arguments"]}}]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_output
                    })
                    # POC phase: collect raw outputs for DS validation
                    _poc_raw_outputs.append({
                        "tool": fn_name,
                        "arguments": fn_args,
                        "output": tool_output,
                    })

                # Continue to next turn
                continue

            # No tool calls — gemma produced the final report
            content = msg.get("content", "")
            latency = time.monotonic() - _t0

            # Extract JSON from markdown code fence
            cleaned = content.strip()
            if "```json" in cleaned:
                json_start = cleaned.find("```json") + 7
                json_end = cleaned.find("```", json_start)
                if json_end != -1:
                    cleaned = cleaned[json_start:json_end].strip()

            # Try to parse as JSON to validate
            try:
                parsed = json.loads(cleaned) if cleaned.strip() else {}
                _log_call(goal, turn_count + 1, latency, True, len(cleaned), "ok")
                return json.dumps({
                    "report": parsed,
                    "model": CRUCIBLE_DEEP_MODEL,
                    "turns": turn_count + 1,
                    "latency_s": round(latency, 1),
                    "poc_raw_outputs": _poc_raw_outputs,
                }, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                # Return raw content if not valid JSON
                _log_call(goal, turn_count + 1, latency, True, len(cleaned), "raw_fallback")
                return json.dumps({
                    "report": cleaned,
                    "model": CRUCIBLE_DEEP_MODEL,
                    "turns": turn_count + 1,
                    "latency_s": round(latency, 1),
                    "poc_raw_outputs": _poc_raw_outputs,
                }, ensure_ascii=False)

        # Hit turn limit
        latency = time.monotonic() - _t0
        _log_call(goal, 15, latency, False, 0, "turn_limit")
        return json.dumps({
            "error": "Crucible exceeded maximum turns (15) without producing a report",
            "report": "",
            "poc_raw_outputs": _poc_raw_outputs,
        }, ensure_ascii=False)

    except Exception as exc:
        latency = time.monotonic() - _t0
        logger.error("crucible_deep failed: %s", exc)
        _log_call(goal, turn_count + 1, latency, False, 0, f"error={exc}")
        return json.dumps({
            "error": f"Crucible execution failed: {exc}",
            "report": "",
            "poc_raw_outputs": _poc_raw_outputs,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------
def check_crucible_requirements() -> bool:
    """Return True if LM Studio is reachable and Gemma is loaded."""
    try:
        import requests as http
        resp = http.get(
            f"{LM_STUDIO_BASE}/api/v0/models",
            timeout=5,
        )
        if resp.status_code != 200:
            return False
        models = resp.json().get("data", [])
        return any("gemma-4-12b" in m.get("id", "") and m.get("state") == "loaded"
                   for m in models)
    except Exception:
        return False


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CRUCIBLE_DEEP_SCHEMA = {
    "name": "crucible_deep",
    "description": (
        "Run a goal through the local Crucible layer (Gemma-4-12b-QAT on LM Studio). "
        "Crucible runs terminal commands and reads files on the local Mac, structures "
        "the findings into clean JSON, and returns the report. DeepSeek never sees "
        "the raw output — only the structured report.\\n\\n"
        "Use this for: system health checks, config audits, log analysis, file "
        "inspections — any task where you want local execution without paying "
        "DeepSeek for terminal tokens."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What to investigate. Be specific about what data you need. "
                    "Examples:\\n"
                    "- 'Check system health: CPU, memory, disk, uptime, services'\\n"
                    "- 'Audit config files in ~/.hermes/profiles for differences'\\n"
                    "- 'Scan gateway logs for errors in the last hour'"
                ),
            },
        },
        "required": ["goal"],
    },
}


# =============================================================================
# Registry
# =============================================================================
from tools.registry import registry

registry.register(
    name="crucible_deep",
    toolset="delegation",
    schema=CRUCIBLE_DEEP_SCHEMA,
    handler=lambda args, **kw: crucible_deep(
        goal=args.get("goal", ""),
    ),
    check_fn=check_crucible_requirements,
    emoji="🔥",
)