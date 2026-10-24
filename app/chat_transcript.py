"""已授权原文转为简单JSON聊天数组，不取数、不推断语义关系或修改权限。"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from app.media_text import inline_media

BEIJING = timezone(timedelta(hours=8))
CHAT_FORMAT = ('聊天记录按原始顺序排列，时间为北京时间；user是作者，role=assistant是你实际说过的话。'
               'id为本地记录ID，message_id为QQ消息号；reply_to是实际QQ引用，in_reply_to是程序记录的机器人回应。'
               '媒体标签为自动说明；本轮只围绕当前任务的新消息交流，旧消息仅作背景。')


def quoted(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


class ChatTranscript:
    def __init__(self, rows: list[dict], runtime: dict):
        self.private = runtime.get('会话类型') == 'private'
        self.by_id = {r['本地记录ID']: r for r in rows if r.get('本地记录ID')}
        self.by_message = {r['消息号']: r for r in rows if r.get('消息号')}
        self.extra = []
        for row in rows:
            for parent in (row.get('引用消息'), row.get('引用或回应原文')):
                if not parent or '内容' not in parent:
                    continue
                if self.lookup(parent) is None:
                    item = dict(parent)
                    self.extra.append(item)
                    if item.get('本地记录ID'):
                        self.by_id[item['本地记录ID']] = item
                    if item.get('消息号'):
                        self.by_message[item['消息号']] = item

    def lookup(self, row):
        return self.by_id.get(row.get('本地记录ID')) or self.by_message.get(row.get('消息号'))

    @staticmethod
    def stamp(value):
        if value is None or value == '':
            return ''
        try:
            stamp = datetime.fromtimestamp(value, BEIJING) if isinstance(value, (int, float)) else datetime.fromisoformat(value)
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=BEIJING)
            return stamp.astimezone(BEIJING).strftime('%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError, OverflowError, OSError):
            return str(value)

    def records(self, rows):
        result = []
        for row in rows:
            qq = row.get('QQ', row.get('作者QQ'))
            name = row.get('昵称', row.get('作者昵称', '用户'))
            bot = row.get('角色') == 'assistant'
            author = '小鲸鱼（你）' if bot else str(name)
            if qq and not (self.private and bot):
                author += f'(QQ{qq})'
            item = {'user': author, 'message': inline_media(row)}
            if row.get('本地记录ID'):
                item['id'] = row['本地记录ID']
            if row.get('时间') is not None:
                item['time'] = self.stamp(row['时间'])
            if row.get('消息号'):
                item['message_id'] = row['消息号']
            if bot:
                item['role'] = 'assistant'
            if row.get('角色') == 'notice' or row.get('事件类型') == '戳一戳':
                item['type'] = 'notice'
            mentions = row.get('@目标QQ') or [r['user_id'] for r in row.get('提到的成员', []) if r.get('user_id')]
            if mentions:
                item['at'] = mentions
            parent = row.get('引用消息') or row.get('引用或回应原文')
            parent_id = row.get('回应消息号') or row.get('引用消息号')
            if parent or parent_id:
                parent = parent or {'消息号': parent_id}
                loaded = self.lookup(parent) or parent
                label = 'in_reply_to' if bot else 'reply_to'
                if loaded.get('本地记录ID'):
                    item[label] = loaded['本地记录ID']
                elif loaded.get('消息号'):
                    item[label + '_message_id'] = loaded['消息号']
            if row.get('原文已截断'):
                item['truncated'] = True
            result.append(item)
        return result

    def render(self, rows):
        return quoted(self.records(rows))


def compact_chat_context(runtime: dict, task: dict, background: dict) -> tuple[dict, dict, dict]:
    working = background.get('短期记忆', {})
    current = task.get('本批新消息', [])
    rows = [*working.get('近期往返', []), *working.get('引用补充', []),
            *background.get('群聊时间线', []), *current]
    if not rows:
        return runtime, task, background
    view = ChatTranscript(rows, runtime)
    unique = {}
    for row in [*rows, *view.extra]:
        key = ('id', row['本地记录ID']) if row.get('本地记录ID') else ('msg', row.get('消息号', id(row)))
        unique.setdefault(key, row)
    rows = sorted(unique.values(), key=lambda r: r.get('本地记录ID', 0))
    runtime = {k: v for k, v in runtime.items() if k != '相关成员索引'}
    runtime['聊天格式'] = CHAT_FORMAT
    if '允许@成员' in runtime:
        runtime['允许@成员'] = [m['QQ'] if isinstance(m, dict) else m for m in runtime['允许@成员']]
    background = {k: v for k, v in background.items() if k not in {'短期记忆', '群聊时间线'}}
    background['聊天记录'] = view.records(rows)
    task = dict(task)
    if current:
        task.pop('本批新消息', None)
        focus = task.pop('当前焦点消息号', None)
        wakeups = task.pop('唤醒消息号', [])
        task['新消息ID'] = [r['本地记录ID'] for r in current if r.get('本地记录ID')]
        focused = next((r for r in current if r.get('消息号') == focus), None) if focus else None
        if focused:
            task['建议焦点ID'] = focused['本地记录ID']
        if wakeups:
            task['唤醒消息ID'] = [r['本地记录ID'] for r in current if r.get('消息号') in wakeups]
    return runtime, task, background
