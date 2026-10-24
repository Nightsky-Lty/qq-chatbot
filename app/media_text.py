"""媒体的有界文本视图；不下载附件，不把识别结果变成用户原话。"""

import json


def inline_media(row: dict) -> str:
    images = row.get('图片资料（非原话）', [])
    by_position = {image.get('位置', i): image for i, image in enumerate(images)}
    used, rendered = set(), []
    # 仅限制派生媒体说明，不截断用户正文。缓存中的完整识别结果仍可回查。
    remaining = 2400

    def picture(position):
        nonlocal remaining
        used.add(position)
        image = by_position.get(position, {})
        name = image.get('图片编号', '无可用编号')
        if image.get('概要状态') != 'ready':
            return f'<image {name} 自动识别={image.get("概要状态", "未解析")} />'
        description = image.get('概要', '')[:240]
        ocr = image.get('图中文字', '')[:240]
        text = f'画面={json.dumps(description, ensure_ascii=False)}'
        if ocr:
            text += f' 图中文字={json.dumps(ocr, ensure_ascii=False)}'
        if image.get('不确定'):
            text += ' 识别不确定'
        if len(text) > remaining:
            return f'<image {name} 自动说明已缩略，细节可按需查看 />'
        remaining -= len(text)
        return f'<image {name} 自动识别，非用户原话>{text}</image>'

    parts = row.get('消息段') or []
    if not parts:
        rendered.append(row.get('内容', ''))
    for part in parts:
        kind = part.get('type')
        if kind == 'text':
            rendered.append(part.get('text', ''))
        elif kind == 'at':
            rendered.append(f'@{part["qq"]}')
        elif kind == 'image':
            rendered.append(picture(part['index']))
        elif kind in {'audio', 'video'}:
            rendered.append(f'<{kind} 内容未解析 />')
        elif kind == 'card':
            # 卡片标题来自消息元数据，绝不冒充已经读取链接正文或看过视频。
            value = {k: part[k] for k in ('platform', 'title', 'description') if part.get(k)}
            rendered.append('<embed 卡片元数据，正文未读取>' + json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '</embed>')
    for position in by_position:
        if position not in used:
            rendered.append(picture(position))
    return ''.join(rendered)
