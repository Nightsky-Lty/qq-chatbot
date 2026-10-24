"""从当前原文提取有界检索线索，不读记忆或数据库。"""

import re


FOLLOWUP = re.compile(r'那[他她它这]|这[个种件些样]|那[个种件些样]|[他她它]呢|刚才|上次|之前|以前|后来|然后|还记得|怎么练|什么颜色'
                      r'|^[他她它](?:的|是|有|会|能|适合|怎么|为什么|叫什么|多少|在哪)|^[这那](?:是|要|能|会|怎么|为什么)')
TOPIC_SWITCH = re.compile(r'换个话题|另外问|另一个问题|先不聊|先别聊|先放一边|先帮我|先看这个|顺便问')
SMALL_TALK = re.compile(r'^(?:你好|嗨|哈[哈啊]*|嗯[嗯哼]*|哦[哦噢]*|好[的吧啊]*|收到|谢谢|晚安|早安|拜拜|[?？!！。\s])+$')
SHORT_FOLLOWUP = re.compile(r'(?:为什么|为啥|怎么说|怎么做|啥意思|什么意思|还有吗|然后呢|怎么练|怎么操作|怎么弄|适合吗|是吗|真的吗|是谁|叫什么|怎么办|后来呢|哪种|哪个|怎么样|咋办)[?？!！。\s]*')


def needs_context(query: str) -> bool:
    text = query.strip()
    if not text or SMALL_TALK.fullmatch(text) or TOPIC_SWITCH.search(text):
        return False
    return len(text) <= 80 and (bool(FOLLOWUP.search(text)) or bool(SHORT_FOLLOWUP.fullmatch(text)))


def focus_background(context, *, history=None, limit=3, assistant_only=False) -> list[str]:
    """父消息优先，其次同批补充和最近往返；群聊回退只取同一作者。"""
    current = context.task.get('本批新消息', [])
    recent = context.working.get('近期往返', []) or context.timeline
    focus_id = context.task.get('当前焦点消息号')
    focus = next((r for r in current if r.get('消息号') == focus_id), current[-1] if current else {})
    visible = [*recent, *current]
    by_id = {r['本地记录ID']: r for r in visible if r.get('本地记录ID')}
    by_message = {r['消息号']: r for r in visible if r.get('消息号')}
    candidates = []
    for parent in (focus.get('引用消息'),):
        if parent:
            candidates.append(by_id.get(parent.get('本地记录ID')) or by_message.get(parent.get('消息号')) or parent)
    candidates.extend(r for r in reversed(current) if r is not focus and r.get('QQ') == focus.get('QQ'))
    candidates.extend(r for r in reversed(recent) if not current or r.get('QQ') == focus.get('QQ') or r.get('角色') == 'assistant')
    if not current and not recent:
        candidates.extend({'内容': r.get('content', ''), '角色': r.get('role')} for r in reversed(history or []))
    result, seen = [], set()
    for row in candidates:
        text = row.get('内容', '').strip()
        if (not text or text in seen or row.get('角色') == 'notice'
                or (assistant_only and row.get('角色') != 'assistant')):
            continue
        seen.add(text)
        result.append(text[:240])
        if len(result) >= limit:
            break
    return result


def memory_context_query(query: str, context, *, history=None) -> str:
    """原请求仍控制日期/回忆意图；此文本只用于补充匹配。"""
    if not needs_context(query):
        return ''
    parts = [text for text in focus_background(context, history=history) if text != query.strip()]
    return '\n'.join([query[:240], *parts]) if parts else ''
