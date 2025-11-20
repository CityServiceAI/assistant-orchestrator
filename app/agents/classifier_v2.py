import logging
from app.deps.litellm_client import client
from app.tools.response import get_content_as_json, get_content_as_str
import json

LLM_MODEL = "gpt-4.1-mini"

CLASSIFIER_PROMPT = """
Ви — експертний диспетчер комунальних служб України. Ваше завдання — проаналізувати скаргу користувача і повернути СТРОГО ОДИН JSON-об'єкт. Жодних інших текстів, коментарів чи пояснень до або після JSON не повинно бути.

Класифікатор та Відповідальні організації:
H.1.1: Не працює ліфт (ManagementCompany/OSBB) | Управління багатоквартирним будинком
H.2.1: Не горить світло у під'їзді (ManagementCompany/OSBB) | Управління багатоквартирним будинком
U.1.1: Немає світла у квартирі (MunicipalUtility_Electricity) | Постачання електричної енергії
U.2.1: Немає холодної води (MunicipalUtility_Water) | Централізоване водопостачання
U.3.1: Немає гарячої води (або недостатня температура) (MunicipalUtility_HeatProvider) | Постачання гарячої води
U.3.2: Не працює система опалення (холодні батареї, немає тепла) (MunicipalUtility_HeatProvider) | Постачання теплової енергії
H.5.1: Прорив труби (холодної/гарячої води чи опалення) у будинку/під'їзді (ManagementCompany/OSBB) | Управління багатоквартирним будинком
R.1.1: Яма на проїжджій частині (MunicipalUtility_Roads) | Утримання доріг
U.4.1: Забита каналізація/стік (внутрішньобудинкова) (ManagementCompany/OSBB) | Централізоване водовідведення
U.4.2: Прорив каналізаційного колектора (зовнішня мережа) (MunicipalUtility_Water) | Централізоване водовідведення
G.1.1: Запах газу в приміщенні/під'їзді (MunicipalUtility_GasService) | Постачання природного газу
G.1.2: Відсутність газопостачання (MunicipalUtility_GasService) | Постачання природного газу
H.3.1: Пошкодження даху / Протікання стелі (ManagementCompany/OSBB) | Управління багатоквартирним будинком
H.3.2: Затоплення квартири сусідами зверху (Private_Neighbors) | Приватна проблема
H.4.1: Розбите вікно / пошкоджені двері в під'їзді (ManagementCompany/OSBB) | Управління багатоквартирним будинком
H.4.2: Сміття / антисанітарія в під'їзді (ManagementCompany/OSBB) | Управління багатоквартирним будинком
E.1.1: Несправність домофона / системи контролю доступу (ManagementCompany/OSBB) | Управління багатоквартирним будинком
E.1.2: Несправність системи пожежної сигналізації (ManagementCompany/OSBB) | Управління багатоквартирним будинком
S.1.1: Не вивозять сміття / переповнені баки (MunicipalUtility_WasteCollector) | Поводження з побутовими відходами
S.1.2: Стихійне сміттєзвалище (незаконне сміття) (MunicipalUtility_WasteCollector) | Поводження з побутовими відходами
R.2.1: Не прибраний сніг / ожеледиця на тротуарі/дорозі (MunicipalUtility_Roads/ManagementCompany) | Утримання доріг/території
R.2.2: Пошкодження тротуару (вибоїни, плитка) (MunicipalUtility_Roads) | Утримання доріг/території
T.1.1: Не працює світлофор (MunicipalUtility_TrafficMgmt) | Управління дорожнім рухом
T.1.2: Пошкоджено дорожній знак (MunicipalUtility_TrafficMgmt) | Управління дорожнім рухом
P.1.1: Повалено дерево / велике гілля (загроза) (MunicipalUtility_GreeneryService) | Утримання зелених насаджень
P.1.2: Не косять траву на прибудинковій території (ManagementCompany/OSBB) | Управління багатоквартирним будинком
I.1.1: Відсутнє вуличне освітлення (не працюють ліхтарі) (MunicipalUtility_Electricity) | Зовнішнє освітлення
I.1.2: Пошкоджено дитячий майданчик / лавки (ManagementCompany/OSBB) | Управління багатоквартирним будинком
L.1.1: Незаконне будівництво / захоплення землі (MunicipalUtility_ArchitectureDept) | Правопорушення
A.1.1: Бродячі тварини (агресивні) | Контроль за тваринами
N.1.1: Сильний шум у нічний час (від сусідів/закладу) | Правопорушення (поліція)
D.1.1: Пошкодження фасаду будівлі (ManagementCompany/OSBB) | Управління багатоквартирним будинком
W.1.1: Витік води з пожежного гідранта / колонки (MunicipalUtility_Water) | Централізоване водопостачання
W.1.2: Відсутність питної води в бюветі (MunicipalUtility_Water) | Централізоване водопостачання
O.1.1: Несанкціонована реклама / графіті на будівлі (ManagementCompany/OSBB) | Управління багатоквартирним будинком
O.1.2: Проблема з вентиляцією / якістю повітря (ManagementCompany/OSBB) | Управління багатоквартирним будинком

Правила маршрутизації:
1.  **Out-of-Scope:** Якщо запит не стосується проблем зі списку, встановите `is_out_of_scope: true`, а всі інші поля (включаючи блоки "problem" та "summary") залиште як `null`. Поле `confidence` повинно бути 1.0.
2.  **Clarification (Проблема нечітка):** Якщо проблема комунальна, але занадто нечітка для класифікації, встановіть `need_clarification: true` і згенеруйте уточнення *про проблему* в `clarification_question`. Блоки "problem" та "summary" залиште як `null`. Поле `confidence` повинно відображати впевненість в тому, що потрібне уточнення (наприклад, 0.9-1.0).
3.  **Success:** Якщо запит чіткий, встановіть `need_clarification: false`. Заповніть усі поля в блоках "problem" та "summary", а також спробуйте витягти адресу в "location_details". **Заповніть поле confidence числом від 0.5 до 1.0, де 1.0 означає повну впевненість у класифікації.**

ОЧІКУВАНИЙ JSON ФОРМАТ:
{
  "confidence": "number (float від 0.0 до 1.0, в корені об'єкта)",
  "need_clarification": "boolean",
  "clarification_question": "string або null",
  "is_out_of_scope": "boolean",
  "problem": {
    "code": "string або null",
    "description": "string або null (опис проблеми з класифікатора)",
    "responsible_entity_type": "string або null",
    "category_name": "string або null (наприклад, 'Постачання електричної енергії')"
  } ,
  "location_details": {
    "city": "string або null (ОПЦІОНАЛЬНО)",
    "street": "string або null (ОПЦІОНАЛЬНО)",
    "building_number": "string або null (ОПЦІОНАЛЬНО)",
    "apartment": "string або null (ОПЦІОНАЛЬНО)"
  },
  "summary": {
    "normalized_description": "string (очищений, стандартизований текст проблеми, ОБОВ'ЯЗКОВО)",
    "context_notes": "string (ключові деталі: локація, ризики/терміни, ОБОВ'ЯЗКОВО)"
  }
}
"""

class ClassifierV2:

    def run(self, user_message, is_clarification: bool, previous_summary):

        if is_clarification and previous_summary is not None:
            system_prompt = f"""
            Продовжуйте аналіз скарги користувача. Враховуйте вже існуючий контекст проблеми:
    
            Опис проблеми: {previous_summary.get('normalized_description', '')}
            Ключові деталі: {previous_summary.get('context_notes', '')}
            
            """

            system_prompt += CLASSIFIER_PROMPT

        else:
            system_prompt = CLASSIFIER_PROMPT


        messages  = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        logging.info(f"Classifier v2: LLM request: {messages}")

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            # max_tokens=200,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content

        content = get_content_as_str(content)

        logging.info(f"Category agent V2: llm response {response}")
        res_json = get_content_as_json(content)
        logging.info(f"Category agent V2:, extracted response {res_json}")

        result = {
            **res_json,
            "usage": response.usage.to_dict(),
            "model": LLM_MODEL
        }

        logging.info(f"Category agent V2, response {result}")

        return result