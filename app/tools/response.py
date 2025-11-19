import json
import logging
import re


def get_content_as_json(raw_content):
    logging.info(f"String to json convertor, raw content: {raw_content}")
    match = re.search(r"```json\n([\s\S]*?)\n```", raw_content)

    if match:
        json_string = match.group(1)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Помилка декодування JSON: {e}")
            return None
    else:
        try:
            return json.loads(raw_content)
        except ValueError as e:
            return None


def get_content_as_str(content):
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text_part = getattr(part, "text", None)
            if isinstance(text_part, str):
                parts.append(text_part)
            elif isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        content = "".join(parts)
    return content.strip() if content is not None else None


def assistant_msg(content):
    return {
        "role": "assistant",
        "content": content,
    }
