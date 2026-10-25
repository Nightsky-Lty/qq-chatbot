"""完整固定角色卡；内容不赋予身份、工具或操作权限。"""

from pathlib import Path


PERSONA_PROMPT = (Path(__file__).with_name('prompts') / 'persona.txt').read_text(encoding='utf-8').strip()
if not PERSONA_PROMPT:
    raise ValueError('角色卡内容不能为空。')
SYSTEM_PROMPT = PERSONA_PROMPT
