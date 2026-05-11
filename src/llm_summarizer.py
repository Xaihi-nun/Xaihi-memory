"""LLM summarizer for xaihi memory system."""
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from .config import config
except ImportError:
    from config import config


DEFAULT_SYSTEM_PROMPT = """You are a memory consolidation assistant. Review the following conversation and generate a structured memory summary.

Important rules:
1. Use actual names/titles from the conversation (e.g., "user", not "the user")
2. summary should be in first person if appropriate, natural tone
3. importance score guidelines:
   - 0.9-1.0: Critical requests, important decisions, strong emotions
   - 0.7-0.8: Clear plans, specific requirements
   - 0.5-0.6: Daily conversations, basic Q&A
   - 0.3-0.4: Casual chat
   - 0.0-0.2: Meaningless
4. Output must be valid JSON format, no other text

# Output Format (JSON)
{
  "summary": "Conversation summary in natural language",
  "topics": ["topic1", "topic2"],
  "key_facts": ["key fact 1", "key fact 2"],
  "sentiment": "positive|neutral|negative",
  "importance": 0.0-1.0
}"""


class LLMSummarizer:
    """Qwen/DashScope LLM summarizer."""

    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(self) -> None:
        cfg = config.get_llm()
        self.model = cfg.get("model", "qwen3.5")
        self.base_url = cfg.get("base_url", "https://dashscope.aliyuncs.com/api/v1")
        self.temperature = cfg.get("temperature", 0.3)

        # Priority: config file > environment variable
        self.api_key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", "DASHSCOPE_API_KEY"))

        if not self.api_key:
            raise ValueError("API key not found in config or environment")

        # Load custom prompt from prompts/summarize_conversation.txt if exists
        # self.system_prompt = self._load_custom_prompt()
        self.system_prompt = f"[Current time: {datetime.now().strftime('%Y-%m-$d %H:%M:%S')}]\n" + self._load_custom_prompt()

    def _load_custom_prompt(self) -> str:
        """Load custom system prompt from prompts/summarize_conversation.txt.

        Falls back to DEFAULT_SYSTEM_PROMPT if file not found.
        """
        prompt_file = Path(__file__).parent.parent / "prompts" / "summarize_conversation.txt"
        if prompt_file.exists():
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return DEFAULT_SYSTEM_PROMPT

    def summarize(self, conversation: str) -> dict[str, Any]:
        """Summarize a conversation and return structured memory."""
        user_prompt = f"""# Conversation Format
[管理员 | timestamp] - 管理员消息
[赛希 | timestamp] - 赛希回复

# Conversation Content
{conversation}

Please output JSON according to the format above."""

        response = self._call_llm(self.system_prompt, user_prompt)
        return self._parse_response(response)

    def incremental_merge(
        self,
        existing: dict[str, Any],
        new_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge new memories into an existing summary.

        Args:
            existing: Current summary dict with summary/topics/key_facts/importance/sentiment.
            new_memories: List of new memory dicts to merge in.

        Returns:
            Updated summary dict.
        """
        existing_text = json.dumps(existing, ensure_ascii=False, indent=2)
        new_items = []
        for i, mem in enumerate(new_memories, 1):
            new_items.append(
                f"- {mem.get('content', '')}\n"
                f"  topics: {mem.get('metadata', {}).get('topics', [])}\n"
                f"  importance: {mem.get('metadata', {}).get('importance', 0.5)}"
            )
        new_text = "\n".join(new_items)

        merge_prompt = self._load_merge_prompt() or (
            "You are a memory consolidation assistant. "
            "Below is an existing summary and some new memories. "
            "Merge the new memories into the summary, preserving all existing information. "
            "Add new topics if needed, enrich existing topics where relevant. "
            "Update importance and sentiment as appropriate. "
            "Output must be valid JSON in the same format."
        )

        user_prompt = f"""## Existing Summary
{existing_text}

## New Memories to Merge
{new_text}

Please output the merged JSON (same format)."""

        response = self._call_llm(merge_prompt, user_prompt)
        return self._parse_response(response)

    def _load_merge_prompt(self) -> str | None:
        """Load merge prompt from prompts/merge_conversation.txt if exists."""
        prompt_file = Path(__file__).parent.parent / "prompts" / "merge_conversation.txt"
        if prompt_file.exists():
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM API (OpenAI-compatible format) with retry."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                    continue
                raise

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM response into structured format."""
        # Try direct JSON parse
        try:
            result = json.loads(response_text)
            return self._validate_and_fill(result)
        except json.JSONDecodeError:
            pass

        # Try to fix common JSON issues
        cleaned = response_text.strip()
        # Remove markdown code blocks
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            result = json.loads(cleaned)
            return self._validate_and_fill(result)
        except json.JSONDecodeError:
            pass

        # Fallback: regex extraction
        summary = re.search(r'"summary"\s*:\s*"([^"]+)"', response_text)
        topics_match = re.findall(r'"topics"\s*:\s*\[([^\]]+)\]', response_text)
        importance_match = re.search(r'"importance"\s*:\s*([0-9.]+)', response_text)
        sentiment_match = re.search(r'"sentiment"\s*:\s*"([^"]+)"', response_text)

        topics = []
        if topics_match:
            topics = [t.strip().strip('"') for t in topics_match[0].split(",")]

        return self._validate_and_fill({
            "summary": summary.group(1) if summary else "Conversation record",
            "topics": topics or ["general"],
            "key_facts": [],
            "sentiment": sentiment_match.group(1) if sentiment_match else "neutral",
            "importance": float(importance_match.group(1)) if importance_match else 0.5,
        })

    def _validate_and_fill(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and fill in missing fields."""
        return {
            "summary": data.get("summary", "Conversation record"),
            "topics": data.get("topics", [])[:5],
            "key_facts": data.get("key_facts", [])[:5],
            "sentiment": data.get("sentiment", "neutral"),
            "importance": float(data.get("importance", 0.5)),
        }


_llm_summarizer_instance = None

def get_llm_summarizer() -> LLMSummarizer:
    """Lazy singleton accessor for LLMSummarizer."""
    global _llm_summarizer_instance
    if _llm_summarizer_instance is None:
        _llm_summarizer_instance = LLMSummarizer()
    return _llm_summarizer_instance
