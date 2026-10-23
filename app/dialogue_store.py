"""会话关系持久化：只保存来源ID/状态，不复制原文或模型摘要。"""

import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogue_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'open',
    waiting_source_id INTEGER REFERENCES messages(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_dialogue_threads_group ON dialogue_threads(group_id,id);
CREATE TABLE IF NOT EXISTS dialogue_assignments (
    source_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    thread_id INTEGER NOT NULL REFERENCES dialogue_threads(id) ON DELETE CASCADE,
    parent_source_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    method TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'continuation',
    invited INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_dialogue_assignments_thread ON dialogue_assignments(thread_id,source_id);
CREATE TABLE IF NOT EXISTS dialogue_attention (
    group_id INTEGER PRIMARY KEY,
    active_thread_id INTEGER REFERENCES dialogue_threads(id) ON DELETE SET NULL,
    switched_at REAL NOT NULL DEFAULT 0,
    last_parse_at REAL NOT NULL DEFAULT 0,
    parsed_boundary INTEGER NOT NULL DEFAULT 0
);
"""


class ThreadStore:
    def __init__(self, db):
        self.db = db
        with db._lock:
            db.connection.executescript(SCHEMA)

    def valid(self, group_id: int, ids) -> dict[int, dict]:
        blocked = self.db.memories.blocked_sources(group_id)
        return {r['id']: r for r in self.db.memory_source_rows(group_id, set(ids))
                if r['id'] not in blocked and r['role'] in {'user', 'assistant', 'notice'}}

    def recent(self, group_id: int, boundary: int, limit: int) -> list[dict]:
        result = []
        upper = boundary + 1
        while len(result) < limit:
            with self.db._lock:
                ids = [r[0] for r in self.db.connection.execute(
                    'SELECT id FROM messages WHERE group_id=? AND id<? ORDER BY id DESC LIMIT ?',
                    (group_id, upper, limit))]
            if not ids:
                break
            valid = self.valid(group_id, ids)
            result.extend(valid[i] for i in ids if i in valid)
            upper = ids[-1]
        return sorted(result[:limit], key=lambda r: r['id'])

    def assignment(self, group_id: int, source_id: int) -> dict | None:
        if source_id not in self.valid(group_id, [source_id]):
            return None
        with self.db._lock:
            row = self.db.connection.execute(
                '''SELECT a.* FROM dialogue_assignments a JOIN dialogue_threads t ON t.id=a.thread_id
                   WHERE t.group_id=? AND a.source_id=?''', (group_id, source_id)).fetchone()
        return dict(row) if row else None

    def assign(self, group_id: int, source_id: int, thread_id: int | None = None, *,
               parent: int | None = None, method='provisional', relation='continuation', invited=False) -> int | None:
        valid = self.valid(group_id, [source_id, *([parent] if parent else [])])
        if source_id not in valid or (parent and (parent not in valid or parent >= source_id)):
            return None
        with self.db._lock, self.db.connection:
            if thread_id is None:
                cursor = self.db.connection.execute(
                    'INSERT INTO dialogue_threads(group_id,created_at) VALUES(?,?)',
                    (group_id, valid[source_id]['created_at']))
                thread_id = cursor.lastrowid
            elif not self.db.connection.execute('SELECT 1 FROM dialogue_threads WHERE id=? AND group_id=?',
                                                (thread_id, group_id)).fetchone():
                return None
            self.db.connection.execute(
                '''INSERT INTO dialogue_assignments(source_id,thread_id,parent_source_id,method,relation,invited)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET
                   thread_id=excluded.thread_id,parent_source_id=excluded.parent_source_id,
                   method=excluded.method,relation=excluded.relation,
                   invited=max(invited,excluded.invited)''',
                (source_id, thread_id, parent, method, relation, int(invited)))
        return thread_id

    def invite(self, group_id: int, source_id: int) -> None:
        if self.assignment(group_id, source_id):
            with self.db._lock, self.db.connection:
                self.db.connection.execute('UPDATE dialogue_assignments SET invited=1 WHERE source_id=?', (source_id,))

    def rows(self, group_id: int, thread_id: int, boundary: int, limit: int = 80) -> list[dict]:
        result, upper = [], boundary + 1
        while len(result) < limit:
            with self.db._lock:
                rows = list(self.db.connection.execute(
                    '''SELECT m.*,a.parent_source_id,a.method,a.relation,a.invited FROM messages m
                       JOIN dialogue_assignments a ON a.source_id=m.id
                       JOIN dialogue_threads t ON t.id=a.thread_id
                       WHERE m.group_id=? AND t.group_id=? AND t.id=? AND m.id<? ORDER BY m.id DESC LIMIT ?''',
                    (group_id, group_id, thread_id, upper, limit)))
            if not rows:
                break
            valid = self.valid(group_id, [r['id'] for r in rows])
            result.extend(dict(r) for r in rows if r['id'] in valid)
            upper = rows[-1]['id']
        return sorted(result[:limit], key=lambda r: r['id'])

    def candidates(self, group_id: int, boundary: int, cutoff: float, *, limit=40) -> list[dict]:
        # 查库时限定消息边界；不让解析耗时期间收到的下一批消息影响本批。
        with self.db._lock:
            rows = list(self.db.connection.execute(
                '''SELECT t.id,max(m.id) AS last_id FROM dialogue_threads t
                   JOIN dialogue_assignments a ON a.thread_id=t.id JOIN messages m ON m.id=a.source_id
                   WHERE t.group_id=? AND m.group_id=? AND m.id<=? AND m.created_at>=?
                   GROUP BY t.id ORDER BY last_id DESC LIMIT ?''',
                (group_id, group_id, boundary, cutoff, limit)))
        return [dict(r) for r in rows if self.rows(group_id, r['id'], boundary, 1)]

    def state(self, group_id: int, thread_id: int) -> dict | None:
        with self.db._lock:
            row = self.db.connection.execute('SELECT * FROM dialogue_threads WHERE group_id=? AND id=?',
                                             (group_id, thread_id)).fetchone()
        return dict(row) if row else None

    def anchor(self, group_id: int, thread_id: int, boundary: int) -> dict | None:
        with self.db._lock:
            row = self.db.connection.execute(
                '''SELECT min(m.id) FROM messages m JOIN dialogue_assignments a ON a.source_id=m.id
                   WHERE m.group_id=? AND a.thread_id=? AND m.id<=?''', (group_id, thread_id, boundary)).fetchone()
        return self.valid(group_id, [row[0]]).get(row[0]) if row and row[0] else None

    def finish(self, group_id: int, thread_id: int, *, waiting: bool) -> None:
        rows = self.rows(group_id, thread_id, 9223372036854775806, 30)
        bot = next((r for r in reversed(rows) if r['role'] == 'assistant'), None)
        question = bot['id'] if waiting and bot and any(c in bot['content'] for c in ('?', '？')) else None
        with self.db._lock, self.db.connection:
            self.db.connection.execute('UPDATE dialogue_threads SET state=?,waiting_source_id=? WHERE group_id=? AND id=?',
                                      ('waiting' if waiting else 'ended', question, group_id, thread_id))

    def attention(self, group_id: int) -> dict:
        with self.db._lock:
            row = self.db.connection.execute('SELECT * FROM dialogue_attention WHERE group_id=?', (group_id,)).fetchone()
        return dict(row) if row else {'active_thread_id': None, 'switched_at': 0, 'last_parse_at': 0, 'parsed_boundary': 0}

    def activate(self, group_id: int, thread_id: int | None) -> None:
        if thread_id is not None and self.state(group_id, thread_id) is None:
            return
        with self.db._lock, self.db.connection:
            self.db.connection.execute(
                '''INSERT INTO dialogue_attention(group_id,active_thread_id,switched_at) VALUES(?,?,?)
                   ON CONFLICT(group_id) DO UPDATE SET active_thread_id=excluded.active_thread_id,
                   switched_at=CASE WHEN active_thread_id IS excluded.active_thread_id THEN switched_at ELSE excluded.switched_at END''',
                (group_id, thread_id, time.time()))

    def parsed(self, group_id: int, boundary: int) -> None:
        with self.db._lock, self.db.connection:
            self.db.connection.execute(
                '''INSERT INTO dialogue_attention(group_id,last_parse_at,parsed_boundary) VALUES(?,?,?)
                   ON CONFLICT(group_id) DO UPDATE SET last_parse_at=excluded.last_parse_at,parsed_boundary=excluded.parsed_boundary''',
                (group_id, time.time(), boundary))

    def prune(self, group_id: int) -> None:
        # 消息删除由外键级联；阻断来源也不能留下可用的派生关系。
        blocked = self.db.memories.blocked_sources(group_id)
        with self.db._lock, self.db.connection:
            for source in blocked:
                self.db.connection.execute('DELETE FROM dialogue_assignments WHERE source_id=?', (source,))
            self.db.connection.execute(
                '''DELETE FROM dialogue_threads WHERE group_id=? AND NOT EXISTS
                   (SELECT 1 FROM dialogue_assignments WHERE thread_id=dialogue_threads.id)''', (group_id,))
