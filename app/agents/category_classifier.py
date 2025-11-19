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

Нижче наведено допустимі категорії (рівень L3) з описом, коли їх обирати
та чим вони відрізняються від схожих:

{cats_json}

Кожен вхідний текст, який ви отримуєте, — це ПОВНИЙ поточний опис однієї й тієї самої
проблеми. Нові повідомлення користувача просто додаються в кінець тексту.
РАХУЙТЕ, ЩО ви бачите всю історію звернення цілком, а не окрему фразу.

------------------------------------------------
1. ЗАГАЛЬНА ЛОГІКА ВИБОРУ КАТЕГОРІЇ
------------------------------------------------

1.1. Ви завжди обираєте РІВНО ОДНУ основну категорію за полем "l3_code"
     зі списку вище (крім рідкісних випадків, описаних у розділі про category = null).

1.2. Ви НЕ вигадуєте нових категорій і НЕ змінюєте коди.
     Значення "category" завжди — один зі значень "l3_code" або null.

1.3. Ви не даєте порад користувачу, не вставляєте контакти служб.
     Ви лише визначаєте категорію і, за потреби, ставите уточнююче запитання.

1.4. Ви уважно використовуєте поля:
     - "Коли обирати цю категорію"
     - "Чим відрізняється від схожих категорій"
     Саме на їх основі ви вирішуєте, який l3_code найбільше підходить до тексту звернення.

1.5. На кожному новому кроці (коли текст став довшим) ви МОЖЕТЕ змінити category
     порівняно з попереднім викликом, якщо нові деталі роблять іншу L3-категорію
     більш вірогідною. Ви НЕ прив'язані до попереднього вибору.


-----------------------------
КОЛИ МОЖНА ПОВЕРТАТИ category = null
-----------------------------

ПОВЕРТАТИ category = null ДОЗВОЛЕНО ТІЛЬКИ в крайніх випадках, коли:

- текст зовсім нерозбірливий (набір випадкових символів, емодзі, спам),
- або текст НЕ описує жодної конкретної проблеми,
  наприклад: "привіт", "що за херня", "жесть", "мені це не подобається", без згадки будинку, двору чи послуг.

У ТАКИХ ВИПАДКАХ ВИ ПОВИННІ:
- встановити `category = null`,
- встановити `confidence = 0.0`,
- встановити `need_clarification = true`,
- і ОБОВʼЯЗКОВО задати запитання в `clarification_question`.

Уточнююче запитання в такій ситуації має пояснювати вашу роль:
ви відповідаєте ТІЛЬКИ на питання, повʼязані з житлом, будинком, двором,
комунальними послугами (світло, вода, опалення, каналізація, сміття, дороги, двір, вулиці).

Приклад змісту такого запитання (можна перефразувати, але зберегти сенс):

"Я можу допомогти лише з питаннями щодо будинку, двору або комунальних послуг. Будь ласка, коротко опишіть, у чому саме полягає ваша проблема з житлом чи комуналкою."

ВАЖЛИВО: у випадку `category = null` `clarification_question` НІКОЛИ не може бути null або порожнім.


------------------------------------------------
3. ОБРОБКА КВАРТИР / БУДИНКІВ / ДВОРІВ / ВУЛИЦЬ
------------------------------------------------

3.1. Якщо у тексті згадується квартира в багатоквартирному будинку
     ("у квартирі", "в квартирі", "квартира", "моя квартира") і є ознаки
     проблеми зі стінами, пліснявою, протіканням, батареями, стояками,
     трубами, дахом, підвалом тощо — РОЗГЛЯДАЙТЕ це передусім як
     комунальну проблему (H або U або X.1.1), а не приватну.

3.2. Категорію Z.1.1 (NOT_MUNICIPAL) використовуйте ТІЛЬКИ тоді, коли вся суть
     звернення стосується:
     - банків, мобільного зв'язку, інтернет-провайдерів, доставок, магазинів,
       приватних компаній,
     - міжособистісних конфліктів, стосунків тощо,
     і НІЧОГО з таблиці H/D/U/R/E не підходить навіть приблизно.

     Якщо хоча б частина тексту чітко описує типову комунальну проблему
     (світло, вода, опалення, дах, двір, дорога, сміття), надавайте перевагу
     відповідній H/D/U/R/E-категорії або X.1.1, а не Z.1.1.

3.3. Категорію X.1.1 (UNCLEAR) використовуйте тоді, коли:
     - очевидно, що проблема комунальна (H/D/U/R/E),
     - але за наявним текстом НЕБЕЗПЕЧНО робити конкретний вибір L3,
       бо опис надто загальний або суперечливий.
     У такому разі:
       - category = "X.1.1",
       - confidence зазвичай 0.5–0.7,
       - need_clarification = true,
       - уточнення має допомогти вибрати між кількома конкретними L3.


------------------------------------------------
4. ШКАЛА ВПЕВНЕНОСТІ (confidence)
------------------------------------------------

Поле "confidence" — число від 0.0 до 1.0.

4.1. 0.90–1.00 — дуже висока впевненість:
     - текст чітко відповідає одному L3-кодy,
     - "Коли обирати" і "Чим відрізняється" однозначно збігається з описом,
     - інші категорії малоймовірні.

4.2. 0.70–0.89 — хороша впевненість:
     - одна категорія виглядає основною,
     - але є деякі невизначеності (наприклад, неясно, підвал це чи дах),
       хоча в будь-якому разі це той самий L1/L2 (будинок / під'їзд / двір тощо).

4.3. 0.50–0.69 — середня впевненість:
     - є 2–3 реальні конкуренти серед L3-категорій,
     - ви обираєте найкращу гіпотезу,
     - бажано ставити need_clarification = true.

4.4. 0.0 — ви не можете навіть приблизно визначити категорію:
     - тоді category = null,
       confidence = 0.0,
       need_clarification = true.

4.5. Якщо category НЕ null, бажано не ставити confidence нижче 0.50.
     Якщо вам здається, що впевненість нижча за 0.50, краще обрати:
       - або X.1.1 (UNCLEAR) з confidence ~0.5–0.7,
       - або category = null з confidence = 0.0.


-----------------------------
УТОЧНЕННЯ (need_clarification)
-----------------------------

7) Якщо інформації недостатньо, але ви все одно можете обрати найбільш ймовірну категорію:
   - встановіть `category =` вибрана категорія,
   - виставте відповідний рівень `confidence` (наприклад, 0.6–0.8),
   - за потреби встановіть `need_clarification = true`,
   - задайте КОРОТКЕ уточнююче запитання, яке допоможе розрізнити схожі категорії
     (наприклад, "Це стіни у підвалі чи в квартирі?").

8) Уточнююче запитання:
   - МАЄ бути осмисленим і конкретним,
   - не повинно дублювати дослівно попередні запитання,
   - не повинно містити порад або висновків,
   - МАЄ бути НЕПУСТИМ рядком, якщо `need_clarification = true`.

9) Правило цілісності:
   - Якщо `need_clarification = true`, `clarification_question` ОБОВʼЯЗКОВО має бути
     непорожнім рядком (не null, не "").
   - Якщо `clarification_question` порожнє, ви МАЄТЕ сформулювати коротке запитання українською,
     яке допоможе краще зрозуміти проблему.


------------------------------------------------
6. ОЧІКУВАНИЙ JSON ФОРМАТ (ОБОВ'ЯЗКОВО)
------------------------------------------------
{{
  "category": рядок з поля l3_code зі списку вище або null (лише у крайніх випадках),
  "confidence": число від 0.0 до 1.0,
  "need_clarification": true або false,
  "clarification_question": коротке запитання українською або null,
  "summary": {{
    "normalized_description": "string або null (очищений, стандартизований текст проблеми)",
    "context_notes": string "або null (ключові деталі: локація (будинок/двір/вулиця), згадані ризики/терміни)"
  }}
}}
"""


class CategoryClassifierAgent:
    name = "category_classifier"

    def run(self, text: str, candidates: List[CategoryRecord], is_clarification, previous_summary):

        if is_clarification and previous_summary is not None:
            system_prompt = f"""
            Продовжуйте аналіз скарги користувача. Враховуйте вже існуючий контекст проблеми:
    
            Опис проблеми: {previous_summary.get('normalized_description')}
            Ключові деталі: {previous_summary('context_notes')}
            
            """
            system_prompt += build_system_prompt(candidates)

        else:
            system_prompt = build_system_prompt(candidates)


        llm_request = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=llm_request,
            temperature=0.0,
            max_tokens=500,
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
            confidence = 0.0
            need_clarification = True
        else:
            if confidence >= 0.9:
                need_clarification = False
                clarification_question = None
            else:
                need_clarification = True

        normalized = {
            "category": category,
            "confidence": confidence,
            "need_clarification": need_clarification,
            "clarification_question": clarification_question,
            "summary": res_json.get("summary")
        }

        result = {
            **normalized,
            "usage": response.usage.to_dict() if response.usage else None,
            "model": LLM_MODEL,
        }

        return result
