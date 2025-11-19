import logging
from app.deps.litellm_client import client
from app.tools.response import get_content_as_json, get_content_as_str

LLM_MODEL = "gpt-4.1-mini"

CLASSIFIER_PROMPT = """
Ви — експертний диспетчер комунальних служб України. Ваше завдання — проаналізувати скаргу користувача і повернути JSON-об'єкт згідно зі схемою.

Класифікатор:
H.1.1: Не працює ліфт (ManagementCompany/OSBB)
H.2.1: Не горить світло у під'їзді (ManagementCompany/OSBB)
U.1.1: Немає світла у квартирі (MunicipalUtility_Electricity)
U.2.1: Немає холодної води (MunicipalUtility_Water)
R.1.1: Яма на проїжджій частині (MunicipalUtility_Roads)

Правила маршрутизації:
1.  **Out-of-Scope (Вимога "a") [31, 32]:** Якщо запит не стосується проблем зі списку (напр., політика, погода, особисті образи, медицина), встанови `is_out_of_scope: true`.
2.  **Clarification (Вимога "б"):** Якщо запит стосується комунальних проблем, але є нечітким (напр., "Все погано", "Немає світла" (де?)), встанови `need_clarification: true` і згенеруй ОДНЕ коротке уточнююче питання в `clarification_question`.
3.  **Success:** Якщо запит чіткий, встанови `need_clarification: false`, `problem_code` та `responsible_entity_type` з таблиці.

Приклади:
- Вхід: "Привіт, як справи?" -> `is_out_of_scope: true`
- Вхід: "вже тиждень не працює ліфт" -> `problem_code: "H.1.1"`, `responsible_entity_type: "ManagementCompany/OSBB"`, `need_clarification: false`
- Вхід: "немає світла" -> `need_clarification: true`, `clarification_question: "Будь ласка, уточніть: світла немає у квартирі чи у під'їзді/на вулиці?"`


JSON формат:
Поверніть ОДИН JSON-об'єкт з такими полями:
- "category": один із code або null
- "confidence": число від 0 до 1
- "need_clarification": true або false
- "clarification_question": рядок або null 
- "is_out_of_scope": true або false
- "responsible_entity_type" відповідно до класифікатор
- "problem_code" відповідно до класифікатор
"""

class ClassifierV2:

    def run(self, messages):

        llm_request  = [{"role": "system", "content": CLASSIFIER_PROMPT}]
        llm_request += messages

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=llm_request,
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content

        content = get_content_as_str(content)

        # logging.info(f"Category agent response {response.model_dump_json(indent=2)}")
        logging.info(f"Category agent V2: llm response {response}")
        res_json = get_content_as_json(content)
        logging.info(f"Category agent V2:, extracted response {res_json}")

        result = {
            **res_json,
            "usage": response.usage.to_dict(),
            "model": LLM_MODEL
        }

        logging.info(f"Category agent, response {result}")

        return result