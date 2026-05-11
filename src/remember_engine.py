"""Remember engine for xaihi memory system."""
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import config
    from .embedding import get_embedding_client
    from .llm_summarizer import get_llm_summarizer
    from .chroma_client import chroma_client
except ImportError:
    from config import config
    from embedding import get_embedding_client
    from llm_summarizer import get_llm_summarizer
    from chroma_client import chroma_client


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def get_buffer_file() -> Path:
    return expand_path(config.get_memory().get("buffer_file", "~/.claude/memory/conversation_buffer.jsonl"))


def get_counter_file() -> Path:
    return expand_path(config.get_memory().get("counter_file", "~/.claude/memory/counter.json"))


def ensure_temp_dir() -> None:
    """Ensure temp directory exists."""
    get_buffer_file().parent.mkdir(parents=True, exist_ok=True)


def read_counter() -> int:
    """Read current round counter."""
    counter_file = get_counter_file()
    if counter_file.exists():
        try:
            with open(counter_file, "r") as f:
                data = json.load(f)
                return data.get("count", 0)
        except Exception:
            return 0
    return 0


def write_counter(count: int) -> None:
    """Write counter value."""
    ensure_temp_dir()
    counter_file = get_counter_file()
    with open(counter_file, "w") as f:
        json.dump({"count": count, "updated_at": datetime.now(timezone.utc).isoformat()}, f)


def append_to_buffer(user_message: str, assistant_message: str) -> None:
    """Append a conversation round to the buffer."""
    ensure_temp_dir()
    buffer_file = get_buffer_file()
    entry = {
        "id": str(uuid.uuid4()),
        "round": read_counter() + 1,
        "user": user_message,
        "assistant": assistant_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(buffer_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_buffer() -> list[dict]:
    """Read all entries from buffer."""
    buffer_file = get_buffer_file()
    if not buffer_file.exists():
        return []

    entries = []
    with open(buffer_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def format_conversation_for_summary(entries: list[dict]) -> str:
    """Format conversation entries for LLM summarization."""
    lines = []
    for entry in entries:
        ts = entry.get("timestamp", "")[:19]
        user = entry.get("user", "").replace("\n", " ").strip()
        assistant = entry.get("assistant", "").replace("\n", " ").strip()
        if user:
            lines.append(f"[管理员 | {ts}] - {user}")
        if assistant:
            lines.append(f"[赛希 | {ts}] - {assistant}")
    return "\n".join(lines)


def clear_buffer() -> None:
    """Clear the conversation buffer."""
    buffer_file = get_buffer_file()
    if buffer_file.exists():
        buffer_file.unlink()


def reset_counter() -> None:
    """Reset counter to zero."""
    write_counter(0)


# ── v2: Ebbinghaus decay ─────────────────────────────────


def calc_effective_importance(
    base_importance: float,
    access_count: int,
    created_at: str,
    now: datetime | None = None,
) -> float:
    """Calculate effective importance using Ebbinghaus forgetting curve.

    adjusted_base = min(1.0, base × (1 + alpha × access_count))
    effective     = adjusted_base × exp(-days_since / S)
    S             = s0 × adjusted_base × (1 + alpha × access_count)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    s0 = config.get("decay.s0", 30)
    alpha = config.get("decay.alpha", 0.15)

    # Parse created_at
    try:
        if isinstance(created_at, str):
            s = created_at.replace("Z", "+00:00")
            created_dt = datetime.fromisoformat(s)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
        else:
            created_dt = created_at
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        created_dt = now

    days_since = (now - created_dt).total_seconds() / 86400.0
    if days_since < 0:
        days_since = 0

    base = max(base_importance, 0.01)
    # access_count can boost the effective base (capped at 1.0)
    adjusted_base = min(1.0, base * (1.0 + alpha * access_count))
    S = s0 * adjusted_base * (1.0 + alpha * access_count)
    effective = adjusted_base * math.exp(-days_since / max(S, 0.001))

    return round(min(effective, 1.0), 4)


def decay_all() -> int:
    """Scan all hot memories and update importance using current decay."""
    all_memories = chroma_client.get_all()
    now = datetime.now(timezone.utc)
    updated = 0

    for mem in all_memories:
        meta = mem.get("metadata", {})
        base = meta.get("base_importance") or meta.get("importance", 0.5)
        access_count = meta.get("access_count", 0)
        created_at = meta.get("created_at", "")

        effective = calc_effective_importance(
            float(base), int(access_count), str(created_at), now
        )

        if effective != meta.get("importance", -1):
            chroma_client.update_metadata(mem["id"], {"importance": effective})
            updated += 1

    return updated


# ── v2: Cold storage ─────────────────────────────────────


def _ensure_cold_dir() -> Path:
    """Ensure cold storage directory exists."""
    cfg_dir = config.get("cold_storage.dir", "~/.claude/memory/cold_storage")
    cold_dir = Path(os.path.expanduser(cfg_dir))
    cold_dir.mkdir(parents=True, exist_ok=True)
    return cold_dir


_llm_call_timestamps: list[float] = []


def _rate_limit(rpm: int | None = None) -> None:
    """Enforce RPM limit on LLM calls."""
    if rpm is None:
        rpm = config.get("settle.rpm_limit", 5)
    global _llm_call_timestamps
    now = time.time()
    cutoff = now - 60.0
    _llm_call_timestamps = [t for t in _llm_call_timestamps if t > cutoff]
    if len(_llm_call_timestamps) >= rpm:
        wait = _llm_call_timestamps[0] + 60.0 - now + 0.1
        if wait > 0:
            time.sleep(wait)
    _llm_call_timestamps.append(time.time())


def write_cold_storage(memories: list[dict], tier: str, date_key: str) -> int:
    """Write memories to cold storage JSONL. Returns number written."""
    cold_dir = _ensure_cold_dir()
    if tier == "daily":
        rel_path = config.get("cold_storage.daily_pattern", "daily/{date}.jsonl")
    else:
        rel_path = config.get("cold_storage.monthly_pattern", "monthly/{month}.jsonl")
    file_name = rel_path.replace("{date}", date_key).replace("{month}", date_key)
    file_path = cold_dir / file_name

    entries = []
    for mem in memories:
        entries.append({
            "id": mem["id"],
            "content": mem.get("content", ""),
            "metadata": mem.get("metadata", {}),
            "archived_at": datetime.now(timezone.utc).isoformat(),
        })

    with open(file_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Update index
    index_file = cold_dir / config.get("cold_storage.index_file", "index.json")
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {}
    for entry in entries:
        index[entry["id"]] = str(file_name)

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return len(entries)


# ── v2: Settlement ───────────────────────────────────────


def _find_merge_target(
    tier: str, date_key: str, candidates: list[dict]
) -> dict | None:
    """Find existing summary in target tier to merge into.

    Priority: 1) date match in target tier  2) embedding similarity > 0.75
    """
    # Try exact date match
    existing = chroma_client.find_by_tier_and_date(tier, date_key)
    if existing:
        return existing[0]

    # Try semantic match
    if candidates:
        query = " ".join(c.get("content", "") for c in candidates)[:500]
        try:
            from .embedding import get_embedding_client

            embedder = get_embedding_client()
            query_emb = embedder.embed(query)
            results = chroma_client.search(query_emb, top_k=3, min_importance=0)
            for r in results:
                if (
                    r["metadata"].get("tier") == tier
                    and r.get("distance", 1.0) < 0.25  # cosine dist → similarity > 0.75
                ):
                    return r
        except Exception:
            pass

    return None


def settle_daily_to_monthly() -> int:
    """Settle low-importance daily memories into monthly summaries."""
    settle_cfg = config.get("settle.daily_to_monthly", {})
    threshold = settle_cfg.get("threshold", 0.35)
    min_candidates = settle_cfg.get("min_candidates", 3)
    shield = config.get("settle.importance_shield", 0.5)
    rpm = config.get("settle.rpm_limit", 5)

    all_memories = chroma_client.get_all()
    now = datetime.now(timezone.utc)

    # Get all daily memories with current effective importance
    daily = [m for m in all_memories if m["metadata"].get("tier") == "daily"]
    if not daily:
        daily = [m for m in all_memories if not m["metadata"].get("tier")]

    # Update effective importance
    for mem in daily:
        meta = mem.get("metadata", {})
        base = meta.get("base_importance") or meta.get("importance", 0.5)
        access_count = meta.get("access_count", 0)
        created_at = meta.get("created_at", "")
        eff = calc_effective_importance(
            float(base), int(access_count), str(created_at), now
        )
        chroma_client.update_metadata(mem["id"], {"importance": eff, "tier": "daily"})
        mem["metadata"]["importance"] = eff
        mem["metadata"]["tier"] = "daily"

    # Group by month
    by_month: dict[str, list[dict]] = {}
    for mem in daily:
        metadata = mem.get("metadata", {})
        created_at = metadata.get("created_at", "")
        month = created_at[:7] if created_at else now.strftime("%Y-%m")
        by_month.setdefault(month, []).append(mem)

    settled = 0
    for month, cands in by_month.items():
        low_eff = [c for c in cands if c["metadata"].get("importance", 0) < threshold]
        high_eff = [c for c in cands if c["metadata"].get("importance", 0) >= shield]

        if len(low_eff) < min_candidates:
            continue

        target = _find_merge_target("monthly", month, low_eff)

        summarizer = get_llm_summarizer()
        existing = {
            "summary": target.get("content", ""),
            "topics": target.get("metadata", {}).get("topics", []),
            "key_facts": target.get("metadata", {}).get("key_facts", []),
            "importance": target.get("metadata", {}).get("base_importance", 0.5),
            "sentiment": target.get("metadata", {}).get("sentiment", "neutral"),
        } if target else None

        _rate_limit(rpm)

        if existing:
            merged = summarizer.incremental_merge(existing, low_eff)
        else:
            merged = summarizer.incremental_merge(
                {"summary": "", "topics": [], "key_facts": [], "importance": 0.5, "sentiment": "neutral"},
                low_eff,
            )

        merged_text = merged.get("summary", "")
        embedding = get_embedding_client().embed(merged_text)

        # Build metadata for monthly entry
        parent_ids = [c["id"] for c in low_eff]
        month_meta = {
            "topics": merged.get("topics", []),
            "key_facts": merged.get("key_facts", []),
            "importance": merged.get("importance", 0.4),
            "base_importance": merged.get("importance", 0.4),
            "sentiment": merged.get("sentiment", "neutral"),
            "source": "settlement",
            "tier": "monthly",
            "created_at": f"{month}-01",
            "access_count": 0,
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "parent_ids": ",".join(parent_ids),
            "merge_count": 1,
            "session_id": f"settle-{month}",
        }

        mem_id = target["id"] if target else f"mem-{uuid.uuid4().hex[:12]}"

        # Move original daily memories to cold storage
        cold_candidates = [m for m in cands if m["id"] not in [c["id"] for c in high_eff]]
        write_cold_storage(cold_candidates, "daily", month)

        # Remove from hot DB
        for mem in low_eff:
            try:
                chroma_client.delete(mem["id"])
            except Exception:
                pass

        # Upsert monthly summary into hot DB
        chroma_client.upsert_memory(mem_id, merged_text, embedding, month_meta)
        settled += 1

    return settled


def settle_monthly_to_yearly() -> int:
    """Settle low-importance monthly memories into yearly summaries."""
    settle_cfg = config.get("settle.monthly_to_yearly", {})
    threshold = settle_cfg.get("threshold", 0.25)
    min_candidates = settle_cfg.get("min_candidates", 2)
    rpm = config.get("settle.rpm_limit", 5)

    all_memories = chroma_client.get_all()
    now = datetime.now(timezone.utc)

    monthly = [m for m in all_memories if m["metadata"].get("tier") == "monthly"]

    # Update effective importance
    for mem in monthly:
        meta = mem.get("metadata", {})
        base = meta.get("base_importance") or meta.get("importance", 0.5)
        access_count = meta.get("access_count", 0)
        created_at = meta.get("created_at", "")
        eff = calc_effective_importance(
            float(base), int(access_count), str(created_at), now
        )
        chroma_client.update_metadata(mem["id"], {"importance": eff})
        mem["metadata"]["importance"] = eff

    # Group by year
    by_year: dict[str, list[dict]] = {}
    for mem in monthly:
        metadata = mem.get("metadata", {})
        created_at = metadata.get("created_at", "")
        year = created_at[:4] if created_at else now.strftime("%Y")
        by_year.setdefault(year, []).append(mem)

    settled = 0
    for year, cands in by_year.items():
        low_eff = [c for c in cands if c["metadata"].get("importance", 0) < threshold]

        if len(low_eff) < min_candidates:
            continue

        target = _find_merge_target("yearly", year, low_eff)

        summarizer = get_llm_summarizer()
        existing = {
            "summary": target.get("content", ""),
            "topics": target.get("metadata", {}).get("topics", []),
            "key_facts": target.get("metadata", {}).get("key_facts", []),
            "importance": target.get("metadata", {}).get("base_importance", 0.5),
            "sentiment": target.get("metadata", {}).get("sentiment", "neutral"),
        } if target else None

        _rate_limit(rpm)

        if existing:
            merged = summarizer.incremental_merge(existing, low_eff)
        else:
            merged = summarizer.incremental_merge(
                {"summary": "", "topics": [], "key_facts": [], "importance": 0.5, "sentiment": "neutral"},
                low_eff,
            )

        merged_text = merged.get("summary", "")
        embedding = get_embedding_client().embed(merged_text)

        parent_ids = [c["id"] for c in low_eff]
        year_meta = {
            "topics": merged.get("topics", []),
            "key_facts": merged.get("key_facts", []),
            "importance": merged.get("importance", 0.3),
            "base_importance": merged.get("importance", 0.3),
            "sentiment": merged.get("sentiment", "neutral"),
            "source": "settlement",
            "tier": "yearly",
            "created_at": f"{year}-01-01",
            "access_count": 0,
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "parent_ids": ",".join(parent_ids),
            "merge_count": 1,
            "session_id": f"settle-{year}",
        }

        mem_id = target["id"] if target else f"mem-{uuid.uuid4().hex[:12]}"

        # Move original monthly to cold storage
        write_cold_storage(low_eff, "monthly", year)

        for mem in low_eff:
            try:
                chroma_client.delete(mem["id"])
            except Exception:
                pass

        chroma_client.upsert_memory(mem_id, merged_text, embedding, year_meta)
        settled += 1

    return settled


def summarize_and_store() -> bool:
    """Summarize buffer content and store to MongoDB."""
    entries = read_buffer()
    if not entries:
        return False

    if len(entries) < 2:
        # Too few entries, don't summarize yet
        return False

    # Format conversation
    conversation = format_conversation_for_summary(entries)

    # Check length limit
    summary_cfg = config.get_summary()
    max_len = summary_cfg.get("max_input_length", 8000)
    if len(conversation) > max_len:
        conversation = conversation[:max_len] + "\n...(对话过长已截断)"

    try:
        # Call LLM to summarize
        result = get_llm_summarizer().summarize(conversation)

        # Generate embedding for the summary
        summary_text = result.get("summary", "")
        embedding = get_embedding_client().embed(summary_text)

        # Generate session ID
        first_entry = entries[0]
        session_id = f"session-{first_entry.get('timestamp', datetime.now(timezone.utc).isoformat())[:10]}-{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()

        # Store in ChromaDB
        chroma_client.add_memory(
            memory_id=f"mem-{uuid.uuid4().hex[:12]}",
            content=summary_text,
            embedding=embedding,
            metadata={
                "topics": result.get("topics", []),
                "key_facts": result.get("key_facts", []),
                "importance": result.get("importance", 0.5),
                "base_importance": result.get("importance", 0.5),
                "sentiment": result.get("sentiment", "neutral"),
                "source": "auto_summary",
                "tier": "daily",
                "access_count": 0,
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "created_at": created_at,
            },
        )

        # Clear buffer and reset counter
        clear_buffer()
        reset_counter()

        return True

    except Exception as e:
        with open("remember_engine_errors.log", "a") as log_file:
            log_file.write(f"{datetime.now().isoformat()} - Error during summarization: {e}\n")
        print(f"Error during summarization: {e}", file=sys.stderr)
        return False


import re


def strip_tool_calls(text: str) -> str:
    """Remove Claude Code tool call blocks from assistant message."""
    if not text:
        return text
    # Remove <tool_use>...</tool_use> blocks
    text = re.sub(r'<tool_use>.*?</tool_use>', '', text, flags=re.DOTALL)
    # Remove <tool-response>...</tool-response> blocks
    text = re.sub(r'<tool-response>.*?</tool-response>', '', text, flags=re.DOTALL)
    # Remove <command-...> blocks
    text = re.sub(r'<command-[^>]*>.*?</command-[^>]*>', '', text, flags=re.DOTALL)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def handle_stop_hook(user_message: str, assistant_message: str) -> None:
    """
    Handle Stop hook: append conversation and check for summarization trigger.
    """
    # Strip tool calls from assistant message
    clean_assistant = strip_tool_calls(assistant_message)
    append_to_buffer(user_message, clean_assistant)

    # Increment counter
    count = read_counter() + 1
    write_counter(count)

    # Check if we should summarize
    trigger_rounds = config.get_memory().get("summary_trigger_rounds", 10)
    if count >= trigger_rounds:
        summarize_and_store()


def handle_session_end() -> None:
    """
    Handle SessionEnd hook: summarize remaining buffer, then run settlement.
    All LLM-heavy settlement work is serial and RPM-limited.
    """
    # 1. Normal summary (existing logic)
    entries = read_buffer()
    if entries:
        summarize_and_store()
    clear_buffer()
    reset_counter()

    # 2. Decay scan
    try:
        decay_all()
    except Exception:
        pass

    # 3. Settlement — daily → monthly → yearly (serial, RPM-limited)
    try:
        settled_daily = settle_daily_to_monthly()
    except Exception:
        settled_daily = 0

    try:
        settled_monthly = settle_monthly_to_yearly()
    except Exception:
        settled_monthly = 0


def handle_session_start() -> int:
    """Decay scan for SessionStart. Fast — math only, no LLM calls.

    Returns number of memories updated.
    """
    return decay_all()


def manual_remember(conversation: str) -> bool:
    """
    Manually trigger a remember operation with given conversation text.
    Used for importing existing memory files.
    """
    if not conversation or len(conversation.strip()) < 10:
        return False

    try:
        result = get_llm_summarizer().summarize(conversation)
        summary_text = result.get("summary", "")
        embedding = get_embedding_client().embed(summary_text)

        session_id = f"manual-{datetime.now(timezone.utc).isoformat()[:10]}-{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()

        chroma_client.add_memory(
            memory_id=f"mem-{uuid.uuid4().hex[:12]}",
            content=summary_text,
            embedding=embedding,
            metadata={
                "topics": result.get("topics", []),
                "key_facts": result.get("key_facts", []),
                "importance": result.get("importance", 0.5),
                "base_importance": result.get("importance", 0.5),
                "sentiment": result.get("sentiment", "neutral"),
                "source": "manual",
                "tier": "daily",
                "access_count": 0,
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "created_at": created_at,
            },
        )
        return True

    except Exception as e:
        print(f"Error during manual remember: {e}", file=sys.stderr)
        return False


def main():
    """Entry point for CLI / hook calls."""
    if len(sys.argv) > 1 and sys.argv[1] == "--session-end":
        handle_session_end()
    elif len(sys.argv) > 1 and sys.argv[1] == "--decay":
        # SessionStart decay scan (math only, fast)
        updated = decay_all()
        print(f"decay: {updated} memories updated")
    elif len(sys.argv) > 2 and sys.argv[1] == "--stop-hook":
        # Called by stop_hook.sh with file path
        tmpfile = sys.argv[2]
        try:
            with open(tmpfile) as f:
                raw = f.read()
            data = json.loads(raw) if raw.strip() else {}

            last_assistant = data.get("last_assistant_message", "")
            transcript_path = data.get("transcript_path", "")

            # Get the last user message from transcript
            last_user = ""
            if transcript_path:
                try:
                    lines = open(transcript_path).readlines()
                    for line in reversed(lines):
                        try:
                            msg = json.loads(line)
                            if msg.get("type") == "user" and not msg.get("isMeta"):
                                content = msg.get("message", {}).get("content", "")
                                if isinstance(content, str) and content.strip():
                                    last_user = content.strip()
                                    break
                                elif isinstance(content, list):
                                    for c in content:
                                        if isinstance(c, dict) and c.get("type") == "text":
                                            last_user = c.get("text", "").strip()
                                            break
                                    if last_user:
                                        break
                        except Exception:
                            continue
                except Exception:
                    pass

            handle_stop_hook(last_user, last_assistant)
        except Exception as ex:
            pass

    elif len(sys.argv) > 1:
        # Fallback: prompt from args
        pass


if __name__ == "__main__":
    main()
