import logging
from app.deps.litellm_client import client
from app.tools.response import get_content_as_json, get_content_as_str
import json

LLM_MODEL = "gpt-4.1-mini"


def get_prompt(summary: dict, user_message: str):
    return f"""
Ви — експертний помічник із вилучення адрес. Ваше завдання — проаналізувати надану інформацію, витягти структуровану адресу, оновити блок summary та визначити, чи достатньо інформації для визначення адреси.

# ПОПЕРЕДНІЙ НОРМАЛІЗОВАНИЙ ОПИС ТА КОНТЕКСТ:
{json.dumps(summary)}

# ПОТОЧНИЙ ТЕКСТ КОРИСТУВАЧА (з новою адресою, якщо є):
{user_message}

# ІНСТРУКЦІЇ:
Поверніть СТРОГО ОДИН JSON-об'єкт, що містить поля "confidence", "need_clarification", "clarification_question", "location_details" та "summary". Не додавайте жодного іншого тексту.

**МІСТО Є ОБОВ'ЯЗКОВИМ ДЛЯ ЗАПОВНЕННЯ, ЯКЩО ВОНО Є В ТЕКСТІ.**
- Якщо адреса достатньо повна (є місто, вулиця, номер будинку) або вже не потребує подальших уточнень, встановіть `need_clarification: false`.
- Якщо інформації про адресу недостатньо для подальшої обробки (наприклад, не вказано місто), встановіть `need_clarification: true` і призначте низький `confidence` (0.5-0.6). У цьому випадку згенеруйте коротке `clarification_question` для запиту повної адреси.

Заповніть `confidence` числом від 0.5 до 1.0, що відображає вашу впевненість у точності вилученої адреси.

ОБОВ'ЯЗКОВО оновіть поле `summary.context_notes`, додавши туди витягнуту адресу (якщо знайдено).

# ОЧІКУВАНИЙ JSON ФОРМАТ:
{{
  "confidence": "number (float від 0.0 до 1.0, впевненість у точності адреси)",
  "need_clarification": "boolean (true, якщо адреса неповна)",
  "clarification_question": "string або null (використовується для запиту повної адреси)",
  "location_details": {{
    "city": "string або null (ОБОВ'ЯЗКОВО, якщо є в тексті)",
    "street": "string або null",
    "building_number": "string або null",
    "apartment": "string або null"
  }},
  "summary": {{
    "normalized_description": "string (очищений, стандартизований текст проблеми/запиту)",
    "context_notes": "string (оновлені ключові деталі: локація, ризики/терміни)"
  }}
}}
"""

class LocationAgent:

    def run(self, summary, user_message):

        messages  = [
            {"role": "system", "content": get_prompt(summary, user_message)},
            {"role": "user", "content": user_message}
        ]

        logging.info(f"Location : LLM request: {messages}")

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            # max_tokens=200,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content

        content = get_content_as_str(content)

        logging.info(f"Location: llm response {response}")
        res_json = get_content_as_json(content)
        logging.info(f"Location: extracted response {res_json}")

        result = {
            **res_json,
            "usage": response.usage.to_dict(),
            "model": LLM_MODEL,
            "agent": "Location"
        }

        logging.info(f"Location, response {result}")

        return result