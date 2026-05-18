"""Recall engine for xaihi memory system."""
import json
import math
import os
import sys
from datetime import datetime
from typing import Any

try:
    from .config import config
    from .embedding import get_embedding_client
    from .chroma_client import chroma_client
    from .remember_engine import calc_effective_importance
except ImportError:
    from config import config
    from embedding import get_embedding_client
    from chroma_client import chroma_client
    from remember_engine import calc_effective_importance


def format_timestamp(dt: Any) -> str:
    """Format datetime to readable string."""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d")
    elif isinstance(dt, str):
        # Try to parse ISO format
        try:
            from datetime import datetime

            dt_obj = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            return dt_obj.strftime("%Y-%m-%d")
        except Exception:
            return dt[:10]
    return str(dt)


def recall(query: str) -> str:
    """
    Main recall function: query memories based on user input.

    Accepts query as argument, or reads JSON from stdin with format:
    {"prompt": "user's message"}

    Outputs formatted memory context to stdout.
    """
    # Parse input: try argument first, then stdin
    if not query:
        try:
            input_data = json.loads(sys.stdin.read())
            if input_data.get("prompt"):
                query = input_data["prompt"]
        except Exception:
            pass

    # Fallback: check command line args
    if not query and len(sys.argv) > 1:
        query = sys.argv[1]

    if not query:
        return ""

    try:
        # Generate embedding for the query
        query_embedding = get_embedding_client().embed(query)

        # Get config
        recall_cfg = config.get_recall()
        top_k = recall_cfg.get("top_k", 5)
        top_candidates = recall_cfg.get("top_candidates", 20)
        min_similarity = recall_cfg.get("min_similarity", 0.65)
        w_similarity = recall_cfg.get("weight_similarity", 0.6)
        w_importance = recall_cfg.get("weight_importance", 0.4)
        max_context_length = recall_cfg.get("max_context_length", 2000)

        # Step 1: Vector search (ChromaDB)
        candidates = chroma_client.search(
            query_embedding, top_k=top_candidates
        )

        if not candidates:
            return ""

        # Step 2: Similarity threshold filtering
        # Quick check: if the best candidate already fails, discard everything
        if candidates[0]["cosine"] < min_similarity:
            return ""

        # Step 3: Calculate recall_score and re-rank
        from datetime import timezone
        now = datetime.now(timezone.utc)
        for c in candidates:
            meta = c.get("metadata", {})
            base = meta.get("base_importance", 0.5)
            ac = meta.get("access_count", 0)
            created_at = meta.get("created_at", "")

            eff = calc_effective_importance(
                float(base), int(ac), str(created_at), now
            )
            c["effective_importance"] = eff
            c["recall_score"] = w_similarity * c["cosine"] + w_importance * eff

        # Sort by recall_score descending
        candidates.sort(key=lambda x: x["recall_score"], reverse=True)

        # Per-memory cosine filter + take top_k
        results = []
        for c in candidates:
            if c["cosine"] < min_similarity:
                break  # Already sorted by recall_score, lower scores will also fail
            results.append(c)
            if len(results) >= top_k:
                break

        if not results:
            return ""

        # Step 4: Update access_count and importance only for passed memories
        for result in results:
            try:
                meta = result.get("metadata", {})
                base = meta.get("base_importance", 0.5)
                ac = meta.get("access_count", 0)
                created_at = meta.get("created_at", "")

                # Calculate new effective importance
                eff = calc_effective_importance(
                    float(base), int(ac), str(created_at), now
                )

                chroma_client.update_metadata(
                    result["id"],
                    {
                        "access_count": int(ac) + 1,
                        "last_accessed": datetime.now().isoformat(),
                        "importance": round(eff, 4),
                    },
                )
            except Exception:
                pass

        # Format results
        format_template = recall_cfg.get(
            "format_template",
            "[相关记忆]\n{memories}\n\n---\n",
        )

        memory_lines = []
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            date_str = format_timestamp(metadata.get("created_at", ""))
            content = result.get("content", "")
            importance = metadata.get("importance", 0)
            topics = ", ".join(metadata.get("topics", [])[:3])

            w1 = w_similarity
            w2 = w_importance
            cos_val = result.get("cosine", 0)
            eff_val = result.get("effective_importance", 0)
            score = result.get("recall_score", 0)

            line = f"- [{date_str}] {content}"
            if topics:
                line += f" (主题：{topics})"
            line += f" ({w1:.1f}×{cos_val:.2f}+{w2:.1f}×{eff_val:.2f}={score:.2f})"
            memory_lines.append(line)

        memories_str = "\n".join(memory_lines)
        output = format_template.format(memories=memories_str)

        # Trim if too long
        if len(output) > max_context_length:
            output = output[:max_context_length] + "\n...(记忆已截断)"

        return output

    except Exception as e:
        return f"<!-- recall error: {e} -->"


if __name__ == "__main__":
    result = recall("")

    print("Current time:", datetime.now().isoformat())
    if result:
        print(result)

    tmp_dir = os.path.expanduser(config.get("memory.temp_dir", "~/.claude/memory"))
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "recall.log"), "wt") as f:
        print("Current time:", datetime.now().isoformat(), file=f)
        print(result, file=f)
