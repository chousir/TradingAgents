"""Gemini CLI-backed client."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from langchain_core.messages import AIMessage

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# ─── Constants (module-level, avoid repeated allocation) ─────────────────────

# Fix #1: Keep only clear patterns that indicate CLI interactive-mode responses.
_META_RESPONSE_PATTERNS = (
    "ready for your first command",
    "i will provide my first command",
    "delegate_to_agent",
    "google_web_search",
    "/tools",
    "awaiting your instruction",
    "what would you like me to",
    "please provide more details",
    "i need more information to proceed",
)

# Fix #2: Move to module scope so bind_tools does not rebuild this set each call.
_RESERVED_TICKER_WORDS = frozenset({
    "SYSTEM", "HUMAN", "TASK", "TOOL", "PHASE",
    "RESULT", "OUTPUT", "DATA", "RESPONSE", "ERROR",
    "ARGS", "CALL", "EXEC", "RUN", "AND", "OR", "NOT",
    "THE", "YOUR", "MY", "OUR", "ITS", "THEIR", "FOR",
    "USE", "BUY", "SELL", "HOLD", "HIGH", "LOW", "OPEN",
    "CLOSE", "FINAL", "REPORT", "TRADE", "STOCK", "FUND",
    "BOND", "CASH", "RISK", "MARKET", "NEWS", "PRICE",
    "WITH", "FROM", "THAT", "THIS", "EACH", "BOTH",
})


# ─── Module-level helper functions ───────────────────────────────────────────

def _normalize_prompt(input_value: Any) -> str:
    """Convert common LangChain input shapes into a single prompt string."""
    if isinstance(input_value, str):
        return input_value

    if isinstance(input_value, (list, tuple)):
        lines: list[str] = []
        for item in input_value:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
            else:
                role = getattr(item, "type", item.__class__.__name__)
                content = getattr(item, "content", str(item))

            role_text = str(role).replace("_", " ").upper()
            lines.append(f"{role_text}: {content}")

        return "\n\n".join(lines)

    if hasattr(input_value, "to_messages"):
        return _normalize_prompt(input_value.to_messages())

    return str(input_value)


def _looks_like_meta_response(text: str) -> bool:
    """Check if the response looks like a CLI meta-response (interactive mode)."""
    lowered = (text or "").lower()
    return any(p in lowered for p in _META_RESPONSE_PATTERNS)


def _extract_response(stdout: str) -> str:
    """Extract the final response from Gemini CLI JSON output."""
    text = stdout.strip()
    if not text:
        return ""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, list):
            parts = []
            for item in response:
                if isinstance(item, dict):
                    part = item.get("text") or item.get("content") or ""
                    if part:
                        parts.append(str(part))
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "".join(parts).strip()

        for key in ("content", "text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()

    return text


def _extract_usage_metadata(stdout: str) -> dict[str, Any] | None:
    """Extract token usage metadata from Gemini CLI JSON output when available."""
    text = stdout.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return None

    models = stats.get("models")
    if not isinstance(models, dict) or not models:
        return None

    first_model_data = next(iter(models.values()), None)
    if not isinstance(first_model_data, dict):
        return None

    tokens = first_model_data.get("tokens")
    if not isinstance(tokens, dict):
        return None

    return {
        "input_tokens": int(tokens.get("input", 0) or 0),
        "output_tokens": int(tokens.get("candidates", 0) or 0),
        "total_tokens": int(tokens.get("total", 0) or 0),
    }


def _extract_ticker(text: str) -> str | None:
    """Extract ticker symbol from text with priority ordering."""
    # Pattern 1: explicit "instrument to analyze is X"
    m = re.search(
        r"instrument to analyze is [`'\"]?([A-Za-z0-9._-]+)[`'\"]?",
        text, re.IGNORECASE,
    )
    if m:
        ticker = m.group(1).upper()
        if ticker not in _RESERVED_TICKER_WORDS:
            return ticker

    # Pattern 2: Taiwan ticker (4-5 digits + .TW/.TWO)
    m = re.search(r"\b(\d{4,5}\.(?:TW|TWO))\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 3: HK ticker (4-5 digits + .HK)
    m = re.search(r"\b(\d{4,5}\.HK)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 4: US ticker (2-6 uppercase letters), excluding reserved words
    for m in re.finditer(r"\b([A-Z]{2,6}(?:\.[A-Z]{1,4})?)\b", text):
        candidate = m.group(1)
        if candidate not in _RESERVED_TICKER_WORDS:
            # Check preceding context to avoid false positives
            start = m.start()
            if start > 0:
                preceding = text[max(0, start - 20):start].lower()
                if any(w in preceding for w in ("the ", "your ", "my ", "our ", "its ")):
                    continue
            return candidate

    return None


def _extract_dates(text: str) -> tuple[str, str]:
    """
    Extract (start_date, end_date) from text.
    Always returns two valid date strings — never None.
    """
    found = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    if not found:
        end_date = today
        start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        return start_date, end_date

    end_date = found[-1]
    if len(found) == 1:
        try:
            start_date = (
                datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)
            ).strftime("%Y-%m-%d")
        except ValueError:
            start_date = end_date
    else:
        start_date = found[0]

    return start_date, end_date


def _truncate_text_by_lines(
    text: str, max_chars: int = 8000, max_lines: int = 200
) -> str:
    """Truncate text by lines instead of hard character cutoff."""
    lines = text.split("\n")
    result_lines: list[str] = []
    char_count = 0

    for line in lines:
        line_len = len(line) + 1
        if char_count + line_len > max_chars or len(result_lines) >= max_lines:
            result_lines.append("... [truncated]")
            break
        result_lines.append(line)
        char_count += line_len

    return "\n".join(result_lines)


def _looks_truncated_report(text: str) -> bool:
    """Detect obvious report truncation or abrupt endings."""
    stripped = (text or "").rstrip()
    if not stripped:
        return True
    if stripped.endswith("... [truncated]"):
        return True

    tail = stripped.splitlines()[-1].strip()
    if not tail:
        return True

    # Handle Markdown formatting in the final line, e.g. "**Risks and Considerations:**"
    # so structural endings are still detected as incomplete.
    tail_plain = re.sub(r"[*_`#>]", "", tail).strip()
    if re.fullmatch(r"[*_`\s]*\*\*[^*]+:\*\*[*_`\s]*", tail):
        return True

    if tail.endswith((":", "-", "•", "/", "(", "[", ",")):
        return True

    if tail_plain.endswith((":", "-", "•", "/", "(", "[", ",")):
        return True

    if tail.lower().startswith(("and ", "or ", "but ", "if ", "to ", "with ")):
        return True

    # Paragraph-like ending without terminal punctuation is usually an incomplete cut.
    # Keep this conservative: require multiple words and exclude common markdown/table lines.
    is_markdown_row = tail_plain.startswith(("|", "- ", "* ", "#"))
    has_terminal_punct = tail_plain.endswith((".", "!", "?"))
    if not is_markdown_row and not has_terminal_punct and len(tail_plain.split()) >= 4:
        return True

    return False


def _has_end_of_report(text: str) -> bool:
    """Check whether the explicit end marker is present."""
    return bool(re.search(r"(?mi)^\s*END OF REPORT\s*$", text or ""))


def _parse_json_array(raw: str) -> list | None:
    """
    Try multiple strategies to extract a JSON array from raw text.
    Fix #3: Use bracket counting instead of greedy regex to handle nested JSON.
    """
    raw = raw.strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown fenced block (non-greedy)
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: bracket counting (Fix #3 - correctly handles nested JSON)
    start = raw.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = raw[start: i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    return None


# ─── Main class ───────────────────────────────────────────────────────────────

class GeminiCliModel:
    """Minimal chat-like wrapper around the Gemini CLI executable."""

    def __init__(
        self,
        model: str,
        command: str = "gemini",
        timeout: int = 600,
        max_retries: int = 2,        # Fix #4: retry for CLI invoke
        retry_delay: float = 3.0,
    ):
        self.model = model
        self.command = command
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.supports_tools = False

    # ── stream ─────────────────────────────────────────────────────────────
    def stream(self, input: Any, config=None, **kwargs) -> Iterator[AIMessage]:
        """Stream fallback — yields the full invoke result as a single chunk."""
        result = self.invoke(input, config=config, **kwargs)
        yield result

    # ── bind_tools ──────────────────────────────────────────────────────────
    def bind_tools(self, *args, **kwargs):
        """
        Emulate LangChain tool calling via 3-phase pipeline:
          Phase 1 — LLM selects tools + args.
          Phase 2 — Python executes selected tools.
          Phase 3 — LLM synthesises final report.
        """
        tools = args[0] if args else []
        tool_list = list(tools or [])
        tool_map: dict[str, Any] = {
            getattr(t, "name", f"tool_{i}"): t
            for i, t in enumerate(tool_list)
        }

        # ── Build tool catalog ────────────────────────────────────────────
        tool_specs: list[str] = []
        for name, tool in tool_map.items():
            description = getattr(tool, "description", "") or ""
            arg_spec = getattr(tool, "args", {}) or {}

            if isinstance(arg_spec, dict):
                if "properties" in arg_spec:
                    props = arg_spec.get("properties", {})
                    required = arg_spec.get("required", [])
                else:
                    props = arg_spec
                    required = list(arg_spec.keys())
            else:
                props = {}
                required = []

            tool_specs.append(
                f"- name: {name}\n"
                f"  description: {description}\n"
                f"  required_args: {required}\n"
                f"  args_schema: {json.dumps(props, ensure_ascii=False)}"
            )
        tool_catalog = "\n".join(tool_specs) if tool_specs else "(no tools)"

        # ── Helpers ───────────────────────────────────────────────────────

        def _infer_tool_args_simple(tool: Any, text: str) -> dict:
            """Fallback arg inference from task text."""
            ticker = _extract_ticker(text)
            start_date, end_date = _extract_dates(text)

            arg_spec = getattr(tool, "args", {}) or {}
            if isinstance(arg_spec, dict):
                if "properties" in arg_spec:
                    props = arg_spec.get("properties", {})
                    required = arg_spec.get("required", [])
                else:
                    props = arg_spec
                    required = list(arg_spec.keys())
            else:
                props = {}
                required = []

            inferred: dict[str, Any] = {}
            for key in props:
                lk = key.lower()
                if lk in ("symbol", "ticker"):
                    inferred[key] = ticker or ""
                elif lk == "query":
                    inferred[key] = ticker or "market analysis"
                elif lk == "start_date":
                    inferred[key] = start_date
                elif lk == "end_date":
                    inferred[key] = end_date
                elif lk in ("curr_date", "current_date"):
                    inferred[key] = end_date
                elif lk == "look_back_days":
                    inferred[key] = 7
                elif lk == "limit":
                    inferred[key] = 5
                elif lk == "freq":
                    inferred[key] = "quarterly"
                elif lk == "indicator":
                    inferred[key] = "rsi"

            for key in required:
                if key not in inferred:
                    inferred[key] = ""

            return inferred

        def _ask_llm_to_select_tools(base_prompt: str, current_date: str) -> list[dict]:
            """Phase 1: Ask Gemini which tools to call and with what args."""
            selection_prompt = (
                "You are a financial analysis assistant. Based on the task below, "
                "decide which tools to call and with what arguments.\n\n"
                f"Current date: {current_date}\n\n"
                f"Available tools:\n{tool_catalog}\n\n"
                f"Task:\n{base_prompt}\n\n"
                "Respond with ONLY a valid JSON array. No explanation or markdown.\n"
                "Each element: {\"name\": \"tool_name\", \"args\": {\"key\": \"value\"}}\n"
                "Rules:\n"
                "1. Only select tools genuinely needed.\n"
                "2. Do NOT repeat the same tool name.\n"
                f"3. Use {current_date} as the reference date.\n\n"
                "Example:\n"
                "[\n"
                f"  {{\"name\": \"get_fundamentals\","
                f" \"args\": {{\"symbol\": \"2330.TW\", \"curr_date\": \"{current_date}\"}}}}\n"
                "]\n\n"
                "JSON array:"
            )

            response = self.invoke(selection_prompt)
            raw = (response.content or "").strip()

            # Fix #3: use _parse_json_array (bracket counting)
            parsed = _parse_json_array(raw)

            if parsed is not None:
                # Fix #5: auto-correct args=null instead of dropping the whole call
                sanitised: list[dict] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    if not isinstance(item.get("name"), str):
                        continue
                    # Auto-correct null / non-dict args
                    if not isinstance(item.get("args"), dict):
                        logger.warning(
                            f"[GeminiCLI] Tool '{item.get('name')}' args is "
                            f"{type(item.get('args')).__name__}, auto-correcting to {{}}."
                        )
                        item["args"] = {}
                    sanitised.append(item)

                if sanitised:
                    return sanitised

            logger.warning(
                f"[GeminiCLI] Tool selection parse failed.\nRaw:\n{raw}\n"
                "Falling back to all tools with inferred args."
            )
            return [
                {"name": name, "args": _infer_tool_args_simple(tool, base_prompt)}
                for name, tool in tool_map.items()
            ]

        def _invoke_tool_with_retry(tool: Any, call_args: dict, max_retries: int = 2) -> Any:
            """Invoke a single tool with retry on transient failure."""
            for attempt in range(max_retries + 1):
                try:
                    if hasattr(tool, "invoke"):
                        return tool.invoke(call_args)
                    elif callable(tool):
                        return tool(**call_args)
                    return "Tool is not invokable"
                except Exception as exc:
                    if attempt < max_retries:
                        logger.warning(
                            f"[GeminiCLI] Tool retry {attempt + 1}/{max_retries}: {exc}"
                        )
                        continue
                    raise

        def _execute_selected_tools(selected: list[dict]) -> list[str]:
            """Phase 2: Execute selected tools with validation, dedup, retry."""
            results: list[str] = []
            # Fix #6: deduplicate by (name, args_hash), allowing same tool with different args
            seen: set[str] = set()

            for call in selected:
                if not isinstance(call, dict):
                    logger.warning(f"[GeminiCLI] Skipping non-dict call: {call}")
                    continue

                name = call.get("name", "")
                call_args = call.get("args", {})

                if not isinstance(name, str) or not name:
                    logger.warning(f"[GeminiCLI] Skipping call with invalid name: {call}")
                    continue

                # Fix #5: auto-correct non-dict args
                if not isinstance(call_args, dict):
                    logger.warning(
                        f"[GeminiCLI] '{name}' args type {type(call_args).__name__},"
                        " auto-correcting to {}."
                    )
                    call_args = {}

                # Fix #6: deduplicate by name + serialized args
                dedup_key = f"{name}:{json.dumps(call_args, sort_keys=True)}"
                if dedup_key in seen:
                    logger.info(f"[GeminiCLI] ⏭️  Duplicate call skipped: {name}")
                    continue
                seen.add(dedup_key)

                if name not in tool_map:
                    logger.warning(f"[GeminiCLI] Unknown tool '{name}'; skipping.")
                    continue

                tool = tool_map[name]
                logger.info(f"[GeminiCLI] 🔧 Executing: {name}({call_args})")

                try:
                    output = _invoke_tool_with_retry(tool, call_args, max_retries=2)
                except Exception as exc:
                    output = f"Tool '{name}' failed after retries: {exc}"
                    logger.error(f"[GeminiCLI] ❌ {name} error: {exc}")

                max_chars = 12000
                max_lines = 250
                if name in ("get_news", "get_global_news"):
                    # News payloads are often long; cap tighter to reduce synthesis token pressure.
                    max_chars = 8000
                    max_lines = 160

                text = _truncate_text_by_lines(str(output), max_chars=max_chars, max_lines=max_lines)
                results.append(
                    f"### Tool Result: {name}\n"
                    f"Arguments: {json.dumps(call_args, ensure_ascii=False)}\n"
                    f"Output:\n{text}"
                )

            return results

        # ── Runner ────────────────────────────────────────────────────────
        def _runner(input_value: Any) -> AIMessage:
            base_prompt = _normalize_prompt(input_value)
            current_date = datetime.utcnow().strftime("%Y-%m-%d")

            if not tool_map:
                no_tools_response = self.invoke(base_prompt)
                no_tools_response.tool_calls = []  # Ensure clean return
                return no_tools_response

            # Phase 1
            logger.info("[GeminiCLI] 🧠 Phase 1: Asking Gemini to select tools...")
            selected_tools = _ask_llm_to_select_tools(base_prompt, current_date)
            logger.info(
                f"[GeminiCLI] 📋 Selected: "
                f"{[t.get('name', '?') for t in selected_tools if isinstance(t, dict)]}"
            )

            # Phase 2
            logger.info("[GeminiCLI] ⚙️  Phase 2: Executing selected tools...")
            tool_results = _execute_selected_tools(selected_tools)

            if not tool_results:
                logger.warning("[GeminiCLI] No tools executed; falling back to direct invoke.")
                fallback_response = self.invoke(base_prompt)
                fallback_response.tool_calls = []  # Ensure clean return
                return fallback_response

            history_text = "\n\n".join(tool_results)

            # Phase 3
            logger.info("[GeminiCLI] 📝 Phase 3: Synthesizing final report...")
            synthesis_prompt = (
                "Write a comprehensive financial analysis report using ONLY the tool results below.\n"
                "Include specific numbers, metrics, and actionable insights.\n"
                "Do NOT mention tool names, infrastructure, or commands.\n"
                "End with a Markdown summary table if relevant.\n"
                "If the report is a final analyst deliverable, end with the exact line: END OF REPORT.\n\n"
                f"Tool Results:\n{history_text}"
            )

            response = self.invoke(synthesis_prompt)
            text = (response.content or "").strip()

            if _looks_like_meta_response(text):
                logger.info("[GeminiCLI] 🔁 Meta-response detected; retrying synthesis...")
                retry_prompt = (
                    "Rewrite as a clean financial analysis report.\n"
                    "Include numbers and insights from the data below.\n"
                    "No tool names, agent internals, or commands.\n\n"
                    f"Tool Results:\n{history_text}"
                )
                retry_response = self.invoke(retry_prompt)
                retry_response.tool_calls = []  # Ensure clean return
                return retry_response

            if _looks_truncated_report(text):
                logger.info("[GeminiCLI] ✂️ Truncated ending detected; asking Gemini to finish cleanly...")
                repair_prompt = (
                    "Finish this report from the last complete thought.\n"
                    "Do not repeat earlier content and do not add new claims.\n"
                    "End with the exact final line: END OF REPORT\n\n"
                    f"Current draft:\n{text}"
                )
                try:
                    repaired = self.invoke(repair_prompt)
                    repaired_text = (repaired.content or "").strip()
                    
                    if repaired_text:
                        logger.info("[GeminiCLI] ✅ Report repair successful; returned complete version")
                        response = repaired
                        text = repaired_text
                    else:
                        logger.warning("[GeminiCLI] ⚠️  Repair returned empty; falling back to original")
                except Exception as e:
                    logger.warning(f"[GeminiCLI] ⚠️  Repair failed with error: {e}; returning original text")

            # Hard completion guard, but only when output still looks incomplete.
            final_text = (response.content or "").strip()
            if final_text and _looks_truncated_report(final_text) and not _has_end_of_report(final_text):
                logger.info("[GeminiCLI] 🧩 Still incomplete after synthesis; requesting one final pass...")
                finalize_prompt = (
                    "Complete this report ending naturally.\n"
                    "Keep existing facts, do not add new claims.\n"
                    "End with the exact final line: END OF REPORT\n\n"
                    f"Draft report:\n{final_text}"
                )
                try:
                    finalized = self.invoke(finalize_prompt)
                    finalized_text = (finalized.content or "").strip()
                    if finalized_text:
                        finalized.tool_calls = []
                        return finalized
                except Exception as e:
                    logger.warning(f"[GeminiCLI] ⚠️  Finalization pass failed: {e}; returning original text")

            # Ensure tool_calls is empty (this is a final report, not a tool invocation request)
            response.tool_calls = []
            return response

        return _runner

    # ── invoke ──────────────────────────────────────────────────────────────
    def invoke(self, input: Any, config=None, **kwargs) -> AIMessage:
        """Invoke Gemini CLI with retry on failure."""  # Fix #4
        prompt = _normalize_prompt(input)
        if not prompt.strip():
            return AIMessage(content="")

        prompt = (
            "Treat this as an isolated one-shot request. "
            "Do not rely on any prior session context.\n\n"
            + prompt
        )

        executable = shutil.which(self.command)
        if not executable:
            raise RuntimeError(
                f"Gemini CLI executable '{self.command}' was not found. "
                "Install Gemini CLI or set GEMINI_CLI_BIN to the correct command."
            )

        cmd = [
            executable,
            "-m", self.model,
            "-p", prompt,
            "--output-format", "json",
        ]

        env = os.environ.copy()
        last_error: Exception | None = None

        # Fix #4: retry loop
        for attempt in range(1, self.max_retries + 2):
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    check=False,
                )

                if completed.returncode != 0:
                    stderr = completed.stderr.strip()
                    stdout = completed.stdout.strip()
                    raise RuntimeError(
                        f"Gemini CLI failed (exit {completed.returncode}).\n"
                        f"STDERR: {stderr or '[empty]'}\n"
                        f"STDOUT: {stdout or '[empty]'}"
                    )

                usage_metadata = _extract_usage_metadata(completed.stdout)
                return AIMessage(
                    content=_extract_response(completed.stdout),
                    usage_metadata=usage_metadata,
                    response_metadata={"gemini_cli_usage": usage_metadata or {}},
                )

            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    logger.warning(
                        f"[GeminiCLI] Attempt {attempt} failed: {exc}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"[GeminiCLI] All {self.max_retries + 1} attempts failed."
                    )

        raise RuntimeError(
            f"Gemini CLI failed after {self.max_retries + 1} attempts."
        ) from last_error


# ─── Client ───────────────────────────────────────────────────────────────────

class GeminiCliClient(BaseLLMClient):
    """Client for the Google Gemini CLI command-line tool."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = "gemini_cli"

    @staticmethod
    def _resolve_cli_model(model: str) -> str:
        """Map preview model aliases to CLI-available model IDs."""
        aliases = {
            "gemini-3-flash-preview": "gemini-2.5-flash",
            "gemini-3.1-flash-lite-preview": "gemini-2.5-flash-lite",
            "gemini-3.1-pro-preview": "gemini-2.5-pro",
        }
        return aliases.get(model, model)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        command = (
            self.kwargs.get("gemini_cli_bin")
            or os.getenv("GEMINI_CLI_BIN")
            or "gemini"
        )
        timeout = int(self.kwargs.get("timeout", 600))
        max_retries = int(self.kwargs.get("max_retries", 2))
        retry_delay = float(self.kwargs.get("retry_delay", 3.0))
        resolved_model = self._resolve_cli_model(self.model)
        return GeminiCliModel(
            resolved_model,
            command=command,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def validate_model(self) -> bool:
        return True
