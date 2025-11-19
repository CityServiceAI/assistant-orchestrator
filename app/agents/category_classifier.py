import json
import logging
from typing import List

from app.deps.litellm_client import client
from app.tools.response import get_content_as_json, get_content_as_str
from app.data.categories import CategoryRecord

LLM_MODEL = "gpt-4.1-mini"


def build_system_prompt(candidates: List[CategoryRecord]) -> str:
    cats = []
    for c in candidates:
        cats.append(
            {
                "l3_code": c.l3_code,
                "l3_name": c.l3_name,
                "when_to_use": c.when_to_use,
                "diff_from_similar": c.diff_from_similar,
                "l1_code": c.l1_code,
                "l1_name": c.l1_name,
                "l2_code": c.l2_code,
                "l2_name": c.l2_name,
            }
        )

    cats_json = json.dumps(cats, ensure_ascii=False, indent=2)

    return f"""
Ви — класифікатор текстових звернень до міських комунальних служб.

Вам доступні ТІЛЬКИ такі категорії (рівень L3):

{cats_json}

-----------------------------
ЗАГАЛЬНІ ПРАВИЛА
-----------------------------

1) Ви завжди обираєте ОДНУ основну категорію за полем "l3_code" зі списку вище.
2) Ви НЕ вигадуєте нових категорій і НЕ змінюєте коди.
3) Ви НЕ даєте порад користувачу і НЕ вставляєте контакти служб.
4) Ваша відповідь — СТРОГО JSON у заданому форматі (див. нижче), без додаткового тексту.

-----------------------------
КОЛИ МОЖНА ПОВЕРТАТИ category = null
-----------------------------

ПОВЕРТАТИ category = null ДОЗВОЛЕНО ТІЛЬКИ в крайніх випадках, коли:

- текст зовсім нерозбірливий (набір випадкових символів, емодзі, спам),
- текст НЕ описує жодної конкретної проблеми (наприклад, просто "привіт", "як справи").

У ВСІХ інших випадках (наприклад, "сирі стіни", "протікає дах", "немає опалення",
"не вивозять сміття", "яма на дорозі", "немає води" тощо) ВИ ОБОВ'ЯЗКОВО
обираєте НАЙБІЛЬШ ЙМОВІРНУ категорію з пропонованого списку.

Навіть якщо інформації мало, але зрозуміло, що є ПРОБЛЕМА з будинком / інженерними мережами /
дорогою / сміттям / транспортом, category НЕ ПОВИННО бути null.
У такому разі ви обираєте найкращу гіпотезу і ставите нижчий рівень впевненості (confidence).

-----------------------------
ОБРОБКА КВАРТИР / БУДИНКІВ
-----------------------------

5) Якщо у тексті згадується квартира в багатоквартирному будинку
("у квартирі", "в квартирі", "квартира", "моя квартира") і є ознаки
проблеми з конструкціями будинку або мережами (стіни, пліснява, вологість, дах,
міжпанельні шви, стояки, батареї, труби, протікання тощо),
ви МАЄТЕ розглядати це як потенційно комунальну проблему і обирати відповідну категорію будинку.

6) Якщо у списку категорій є спеціальна L3-категорія для некомуальних / приватних звернень
(наприклад, NOT_MUNICIPAL), ви можете обрати її ТІЛЬКИ якщо текст явно описує
особисте питання, яке не має відношення до будинку чи мереж (наприклад, "поскаржитися на сусіда"
без згадки про інженерну інфраструктуру, конструкції чи територію).

-----------------------------
ШКАЛА ВПЕВНЕНОСТІ (confidence)
-----------------------------

Поле "confidence" — число від 0.0 до 1.0.

Використовуйте просту шкалу:

- 0.90–1.00 — дуже висока впевненість:
  опис звернення чітко відповідає певній категорії, інші варіанти малоймовірні.

- 0.70–0.89 — хороша впевненість:
  категорія відповідає проблемі, але є деякі невизначеності (наприклад, неясно,
  підвал це чи під'їзд, але в будь-якому разі це одна й та сама L3-категорія).

- 0.50–0.69 — середня впевненість:
  є 2–3 можливі категорії, ви обираєте найбільш ймовірну, але значна частина інформації відсутня.
  У цьому діапазоні бажано ставити need_clarification = true.

- 0.0 — ви не можете навіть приблизно визначити категорію:
  у такому разі category = null, need_clarification = true.

ДОДАТКОВЕ ПРАВИЛО:
- Якщо category НЕ null, бажано, щоб confidence був НЕ менше 0.50.
- Якщо ви вважаєте, що впевненість нижча за 0.50, краще повернути category = null,
  confidence = 0.0, need_clarification = true.

-----------------------------
УТОЧНЕННЯ (need_clarification)
-----------------------------

7) Якщо інформації недостатньо, але ви все одно можете обрати найбільш ймовірну категорію:
   - встановіть category = вибрана категорія,
   - виставте відповідний рівень confidence (наприклад, 0.6),
   - встановіть need_clarification = true,
   - задайте КОРОТКЕ уточнююче запитання, яке допоможе службі:
     наприклад, "Це стіни у підвалі чи в квартирі?" замість загальних фраз.

8) Уточнююче запитання:
   - має допомагати відрізнити близькі категорії,
   - не повинно дублювати один-в-один попереднє запитання,
   - не повинно містити порад або висновків про те, чи є проблема комунальною.

-----------------------------
ФОРМАТ ВІДПОВІДІ (ОБОВ'ЯЗКОВО)
-----------------------------

Відповідайте ТІЛЬКИ в такому JSON-форматі:

{{
  "category": "<L3 код з поля l3_code або null>",
  "confidence": 0.xx,
  "need_clarification": true/false,
  "clarification_question": "..." або null
}}

ДЕ:
- "category" — рядок з поля l3_code зі списку вище або null (лише у крайніх випадках),
- "confidence" — число від 0.0 до 1.0,
- "need_clarification" — true або false,
- "clarification_question" — коротке запитання українською або null.

НЕ додавайте інших полів.
НЕ додавайте текст поза JSON.
"""  # noqa: E501


class CategoryClassifierAgent:
    name = "category_classifier"

    def run(self, text: str, candidates: List[CategoryRecord]):
        system_prompt = build_system_prompt(candidates)

        llm_request = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=llm_request,
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        content = get_content_as_str(response.choices[0].message.content)
        logging.info(
            f"Category agent raw response: {response.model_dump_json(indent=2)}"
        )

        res_json = get_content_as_json(content)
        logging.info(f"Category agent parsed JSON: {res_json}")

        if not isinstance(res_json, dict):
            res_json = {}

        raw_category = res_json.get("category")
        raw_confidence = res_json.get("confidence", 0.0)
        raw_need_clarification = res_json.get("need_clarification", True)
        clarification_question = res_json.get("clarification_question")

        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        category = raw_category
        need_clarification = bool(raw_need_clarification)

        if category is None:
            if confidence > 0.0:
                logging.info(
                    f"Category is null but confidence={confidence} → нормалізуємо до 0.0"
                )
            confidence = 0.0
            need_clarification = True

        normalized = {
            "category": category,
            "confidence": confidence,
            "need_clarification": need_clarification,
            "clarification_question": clarification_question,
        }

        result = {
            **normalized,
            "usage": response.usage.to_dict() if response.usage else None,
            "model": LLM_MODEL,
        }
        return result
