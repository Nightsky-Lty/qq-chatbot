"""单次Agent任务的工作区；长期整合只引用真实新证据，不持久化推理/工具轨迹。"""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    namespace: int
    source_ids: set[int]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict[str, Any]] = field(default_factory=list)
    search_cache: dict = field(default_factory=dict)
    closed: bool = False

    def release(self):
        self.messages.clear()
        self.search_cache.clear()
        self.source_ids.clear()
        self.closed = True
