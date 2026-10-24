"""长期记忆的模型视图；完整记录、检索和管理工具接口保持结构化。"""

from datetime import datetime

from app.chat_transcript import BEIJING, quoted


MEMORY_FORMAT = '记忆#为记忆编号，v为版本，原话#/来源为聊天记录ID；时间为北京时间。原话和旧记忆仅作背景，依据、条件和不确定性不能省略或当成已核实事实。'
LAYERS = {'event': '事件', 'abstract': '抽象', 'perceptual': '图片', 'source': '原话'}
FIELDS = {'依据': '依据', '有效期': '有效至', 'conditions': '条件', 'uncertain': '不确定',
          '来源记录': '来源', '来源摘录': '来源摘录', '来源说明': '来源说明', '图片编号': '图片',
          '关联图片': '关联图片', '关联记忆': '关联记忆', 'participants': '参与者', 'aliases': '别名',
          'media_ref_id': '图片引用'}


def stamp(value):
    try:
        return datetime.fromtimestamp(value, BEIJING).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError, OverflowError, OSError):
        return quoted(value)


def compact_memories(items: list[dict]) -> str | list[dict]:
    # 兼容显式调用提供的其他资料形状，不丢弃未知字段。
    if not any('记忆编号' in item or item.get('层') in LAYERS for item in items):
        return items
    lines = []
    for item in items:
        parts = []
        if '记忆编号' in item:
            parts.append(f"记忆#{item['记忆编号']}")
        elif item.get('层') == 'source' and '本地记录ID' in item:
            parts.append(f"原话#{item['本地记录ID']}")
        else:
            parts.append('资料')
        if '版本' in item:
            parts.append(f"v{item['版本']}")
        if item.get('层'):
            parts.append(LAYERS.get(item['层'], quoted(item['层'])))
        if item.get('成员QQ') and not (item.get('层') == 'source' and item.get('角色') == 'assistant'):
            parts.append(f"QQ{item['成员QQ']}")
        if item.get('角色'):
            parts.append('角色=' + quoted(item['角色']))
        if item.get('发生时间'):
            parts.append('发生=' + stamp(item['发生时间']))
        elif item.get('记录时间'):
            parts.append('记录=' + stamp(item['记录时间']))
        parts.append('内容=' + quoted(item.get('内容', '')))
        for field, label in FIELDS.items():
            if field == '有效期' and not item.get(field):
                continue
            if field in item and item[field] is not None and item[field] != [] and item[field] != '':
                value = stamp(item[field]) if field == '有效期' else quoted(item[field])
                parts.append(label + '=' + value)
        known = {'记忆编号', '本地记录ID', '版本', '层', '成员QQ', '角色', '发生时间', '记录时间', '内容', *FIELDS}
        extra = {k: v for k, v in item.items() if k not in known}
        if extra:
            parts.append('补充=' + quoted(extra))
        lines.append(' '.join(parts))
    return '\n'.join(lines)
