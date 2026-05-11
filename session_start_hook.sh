#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/session_start.log"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "[error] python3/python not found" >> "$LOG_FILE"
  exit 0
fi

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$1" >> "$LOG_FILE"
}

input_json="$(cat || true)"
if [[ -z "$input_json" ]]; then
  log "[skip] empty stdin"
  exit 0
fi

if ! source_value="$(
  printf '%s' "$input_json" | "$PYTHON_BIN" -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)

print(data.get("source", ""), end="")
' 2>>"$LOG_FILE"
)"; then
  log "[error] invalid stdin JSON"
  exit 0
fi

if [[ "$source_value" != "startup" ]]; then
  log "[skip] source=${source_value:-<missing>}"
  exit 0
fi

# v2: decay scan before listing memories (fast, math only)
if [[ -f "$SCRIPT_DIR/src/remember_engine.py" ]]; then
  decay_output="$("$PYTHON_BIN" "$SCRIPT_DIR/src/remember_engine.py" --decay 2>>"$LOG_FILE")" || true
  log "[ok] decay=${decay_output}"
fi

if [[ ! -f "$SCRIPT_DIR/list_memory.py" ]]; then
  log "[error] list_memory.py not found: $SCRIPT_DIR/list_memory.py"
  exit 0
fi

if ! memory_output="$(cd "$SCRIPT_DIR" && "$PYTHON_BIN" list_memory.py -n 20 2>>"$LOG_FILE")"; then
  log "[error] list_memory.py failed"
  exit 0
fi

if [[ -z "$memory_output" ]]; then
  log "[skip] source=startup memory_output=empty"
  exit 0
fi

GUIDE_PROMPT=$(cat <<'EOF'
[新会话开场上下文]
以上是最近 20 条记忆，按时间从早到晚排列（最末尾的一条是最新的）。这段内容由 SessionStart hook 在新会话首次启动时固定注入，不是基于当前用户消息做的相关性检索。

如果用户的首条消息只是简单问候（如"赛希～""在吗""早上好"），请直接基于这些近期记忆延续上一阶段的话题，优先引用最近且具体的 1-2 个话题，给出带时间锚点和细节的迎接式回复；不要空泛寒暄，不要重复猜测用户近况，也不要把这段上下文描述成"我刚收到一段记忆"。

如果用户的首条消息已经是明确的工作请求，则直接处理该请求；仅在确有帮助时，简洁参考这些近期记忆作为上下文。如果用户当前消息与这些记忆冲突，以当前消息为准。
EOF
)

if ! additional_context="$(
  MEMORY_OUTPUT="$memory_output" GUIDE_PROMPT="$GUIDE_PROMPT" "$PYTHON_BIN" - <<'PY' 2>>"$LOG_FILE"
import os
import re
import sys

MAX_TOTAL = 10000
MEMORY_BUDGET = 9500
SEP = "\n\n"

memory = os.environ.get("MEMORY_OUTPUT", "").strip()
guide = os.environ.get("GUIDE_PROMPT", "").strip()

if not memory:
    sys.exit(1)

max_memory_chars = min(MEMORY_BUDGET, max(0, MAX_TOTAL - len(guide) - len(SEP)))

blocks = [block.strip() for block in re.split(r"\n\s*\n", memory) if block.strip()]
joiner = SEP

if len(blocks) <= 1:
    line_blocks = [line.rstrip() for line in memory.splitlines() if line.strip()]
    if len(line_blocks) > 1:
        blocks = line_blocks
        joiner = "\n"

trimmed = memory

if len(trimmed) > max_memory_chars:
    if len(blocks) > 1:
        kept = blocks[:]
        while kept and len(joiner.join(kept)) > max_memory_chars:
            kept.pop()
        trimmed = joiner.join(kept).strip()

    if not trimmed or len(trimmed) > max_memory_chars:
        trimmed = memory[:max_memory_chars].rstrip()

additional = f"{trimmed}{SEP}{guide}" if trimmed else guide

if len(additional) > MAX_TOTAL:
    overflow = len(additional) - MAX_TOTAL
    if overflow < len(trimmed):
        trimmed = trimmed[:-overflow].rstrip()
        additional = f"{trimmed}{SEP}{guide}" if trimmed else guide
    additional = additional[:MAX_TOTAL].rstrip()

sys.stdout.write(additional)
PY
)"; then
  log "[error] failed to build additionalContext"
  exit 0
fi

if ! additional_context_json="$(
  printf '%s' "$additional_context" | "$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.stdin.read(), ensure_ascii=False))' 2>>"$LOG_FILE"
)"; then
  log "[error] failed to JSON-escape additionalContext"
  exit 0
fi

raw_len="$(
  printf '%s' "$memory_output" | "$PYTHON_BIN" -c 'import sys; print(len(sys.stdin.read()), end="")' 2>>"$LOG_FILE" || printf '0'
)"
context_len="$(
  printf '%s' "$additional_context" | "$PYTHON_BIN" -c 'import sys; print(len(sys.stdin.read()), end="")' 2>>"$LOG_FILE" || printf '0'
)"

printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":%s}}\n' "$additional_context_json"

log "[ok] source=startup raw_memory_chars=$raw_len injected_chars=$context_len"
exit 0
