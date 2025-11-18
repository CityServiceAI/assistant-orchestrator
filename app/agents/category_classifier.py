import json
import re
from enum import Enum
import logging
from app.deps.litellm_client import client

LLM_MODEL = "gpt-4.1-mini"


class CategoryCode(str, Enum):
    WATER_SUPPLY = "WATER_SUPPLY"
    HEATING = "HEATING"
    ELECTRICITY = "ELECTRICITY"
    ROAD_INFRASTRUCTURE = "ROAD_INFRASTRUCTURE"
    WASTE = "WASTE"
    YARD_TERRITORY = "YARD_TERRITORY"
    PUBLIC_TRANSPORT = "PUBLIC_TRANSPORT"
    OTHER = "OTHER"
    NOT_MUNICIPAL = "NOT_MUNICIPAL"


CATEGORIES = [
    {
        "code": "WATER_SUPPLY",
        "title_uk": "Проблеми з водопостачанням",
        "description_uk": "Відсутність води, низький тиск, іржава вода, прориви труб, витік води тощо.",
    },
    {
        "code": "HEATING",
        "title_uk": "Опалення",
        "description_uk": "Відсутність опалення, холодні батареї, протікання системи опалення.",
    },
    {
        "code": "ELECTRICITY",
        "title_uk": "Електропостачання",
        "description_uk": "Відключення світла, аварійна електропроводка, мерехтіння світла.",
    },
    {
        "code": "ROAD_INFRASTRUCTURE",
        "title_uk": "Дороги та тротуари",
        "description_uk": "Ями, погане покриття, зруйновані тротуари.",
    },
    {
        "code": "WASTE",
        "title_uk": "Сміття та прибирання",
        "description_uk": "Переповнені баки, несвоєчасне вивезення сміття, стихійні сміттєзвалища.",
    },
    {
        "code": "YARD_TERRITORY",
        "title_uk": "Прибудинкова територія",
        "description_uk": "Освітлення двору, дитячі майданчики, дерева, лавочки тощо.",
    },
    {
        "code": "PUBLIC_TRANSPORT",
        "title_uk": "Громадський транспорт",
        "description_uk": "Маршрути, графік руху, зупинки, стан транспорту.",
    },
    {
        "code": "OTHER",
        "title_uk": "Інше (комунальна тема)",
        "description_uk": "Комунальні проблеми, які не підпадають під інші категорії.",
    },
    {
        "code": "NOT_MUNICIPAL",
        "title_uk": "Не стосується комунальних послуг",
        "description_uk": "Текст не про комунальні послуги або міську інфраструктуру.",
    },
]

CATEGORIES_JSON = json.dumps(CATEGORIES, ensure_ascii=False, indent=2)

SYSTEM_CATEGORY_PROMPT = f"""
Ви — класифікатор звернень до міських комунальних служб.

Вхід: офіційний, нейтральний текст звернення українською мовою.

Доступні категорії (code):
{CATEGORIES_JSON}

Правила:
1) Виберіть одну категорію "code" з наведених.
2) Якщо звернення не стосується комунальних послуг або міської інфраструктури —
   використайте "NOT_MUNICIPAL".
3) Якщо інформації недостатньо, але тема комунальна —
   оберіть найбільш ймовірну категорію, встановіть need_clarification=true
   і сформуйте коротке уточнююче запитання.
4) Якщо текст зовсім незрозумілий — category = null, confidence = 0.0, need_clarification = true.

Формат відповіді СТРОГО:
Поверніть ОДИН JSON-об'єкт з такими полями:
- "category": один із code або null
- "confidence": число від 0 до 1
- "need_clarification": true або false
- "clarification_question": рядок або null

НЕ додавайте жодних пояснень поза JSON.
"""


class CategoryClassifierAgent:
    name = "category_classifier"

    def run(self, messages: list[dict], next_message):
        """
        Вхід: вже очищений та нормалізований український текст (мовним агентом).
        Вихід: CategoryDetectionResult з полями category/confidence/need_clarification/clarification_question.
        """

        llm_request  = [{"role": "system", "content": SYSTEM_CATEGORY_PROMPT}]
        llm_request += messages
        llm_request += [{"role": "user", "content": next_message}]

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=llm_request,
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content

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

        logging.info(f"Category agent response {response.model_dump_json(indent=2)}")
        res_json = self.get_content_as_json(content)
        logging.info(f"Category agent, extracted response {res_json}")

        result = {
            **res_json,
            "usage": response.usage.to_dict(),
            "model": LLM_MODEL
        }

        logging.info(f"Category agent, response {result}")
        return result

    @staticmethod
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
