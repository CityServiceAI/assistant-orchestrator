import json
import re
from datetime import datetime

import yaml


def get_content_as_json(raw_content):
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


def assistant_msg(content, agent=None):
    return {
        "role": "assistant",
        "content": content,
        "agent": agent
    }


def get_current_date_info():
    """
    Генерує рядок з поточною датою та порою року українською мовою.
    """

    # Отримання поточної дати та часу
    now = datetime.now()

    # Форматування дати: "21 листопада 2025 року"
    # %d - день місяця
    # %B - назва місяця (локалізована)
    # %Y - рік
    date_str = now.strftime("%d %B %Y року")

    # Визначення пори року на основі місяця
    month = now.month
    if 3 <= month <= 5:
        season = "весна"
    elif 6 <= month <= 8:
        season = "літо"
    elif 9 <= month <= 11:
        season = "осінь"
    else:
        season = "зима"

    # Комбінування інформації у фінальний рядок
    info_string = f"{date_str}. Зараз {season}."

    return info_string


def load_config(file_path):
    """
    Завантажує вміст YAML-файлу в Python-словник.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Використовуємо SafeLoader для безпечного парсингу
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        print(f"Помилка: Файл конфігурації '{file_path}' не знайдено.")
        return None
    except yaml.YAMLError as e:
        print(f"Помилка парсингу YAML-файлу: {e}")
        return None
