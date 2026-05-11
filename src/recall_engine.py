"""Recall engine for xaihi memory system."""
import json
import os
import sys
from datetime import datetime
from typing import Any

try:
    from .config import config
    from .embedding import get_embedding_client
    from .chroma_client import chroma_client
except ImportError:
    from config import config
    from embedding import get_embedding_client
    from chroma_client import chroma_client


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
        memory_cfg = config.get_memory()
        recall_cfg = config.get_recall()
        top_k = memory_cfg.get("top_k", 5)
        min_importance = memory_cfg.get("min_importance", 0.3)

        # Search ChromaDB
        results = chroma_client.search(
            query_embedding, top_k=top_k, min_importance=min_importance
        )

        if not results:
            return ""

        # v2: update access tracking for hit memories
        for result in results:
            try:
                meta = result.get("metadata", {})
                new_access_count = meta.get("access_count", 0) + 1
                chroma_client.update_metadata(
                    result["id"],
                    {
                        "access_count": new_access_count,
                        "last_accessed": datetime.now().isoformat(),
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

            line = f"- [{date_str}] {content}"
            if topics:
                line += f" (主题: {topics})"
            line += f" (重要性: {importance:.1f})"
            memory_lines.append(line)

        memories_str = "\n".join(memory_lines)
        output = format_template.format(memories=memories_str)

        # Trim if too long
        max_len = recall_cfg.get("max_context_length", 2000)
        if len(output) > max_len:
            output = output[:max_len] + "\n...(记忆已截断)"

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
        print(result, file=f)
    