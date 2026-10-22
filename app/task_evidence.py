"""单次Agent的证据与查询缓存；不是对话短期记忆，不存内部推理。"""

import uuid
from dataclasses import dataclass, field


@dataclass
class TaskEvidence:
    namespace: int
    source_ids: set[int]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    new_source_ids: set[int] = field(default_factory=set)
    search_cache: dict = field(default_factory=dict)
    closed: bool = False

    def release(self):
        self.search_cache.clear()
        self.source_ids.clear()
        self.new_source_ids.clear()
        self.closed = True
