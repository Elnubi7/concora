from __future__ import annotations

from collections import defaultdict, deque
from uuid import uuid4


class SessionStore:
    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns * 2))
        self._assistant_turns: dict[str, int] = defaultdict(int)

    def ensure_conversation(self, conversation_id: str | None = None) -> str:
        if conversation_id and conversation_id.strip():
            return conversation_id.strip()
        return f"conv_{uuid4().hex[:16]}"

    def get_history(self, conversation_id: str) -> list[dict]:
        return list(self._history.get(conversation_id, []))

    def get_recent_user_messages(self, conversation_id: str, limit: int = 2) -> list[str]:
        messages = [item["content"] for item in self._history.get(conversation_id, []) if item["role"] == "user"]
        return messages[-limit:]

    def get_turn_number(self, conversation_id: str) -> int:
        return self._assistant_turns.get(conversation_id, 0)

    def append_user_message(self, conversation_id: str, content: str) -> None:
        self._history[conversation_id].append({"role": "user", "content": content})

    def append_assistant_message(self, conversation_id: str, content: str) -> int:
        self._history[conversation_id].append({"role": "assistant", "content": content})
        self._assistant_turns[conversation_id] += 1
        return self._assistant_turns[conversation_id]

    def clear(self, conversation_id: str) -> None:
        self._history.pop(conversation_id, None)
        self._assistant_turns.pop(conversation_id, None)
