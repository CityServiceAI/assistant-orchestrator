import json
import logging
import os

import app.data.categories_faiss as categories
# import app.data.categories_chromadb as categories
from app.deps.litellm_client import client
from app.tools.response import get_content_as_json, get_content_as_str, get_current_date_info, load_config

CLASSIFIER_V3_PROMPT = os.getenv("CLASSIFIER_V3_PROMPT", "app/agents/classifier_v3.yaml")
AGENT_CONFIG = load_config(CLASSIFIER_V3_PROMPT)

class ClassifierV3:

    def run(self, state):

        user_message = state["messages"][-1]['content']
        is_clarification = state.get("need_clarification")
        previous_summary = state.get('summary', {})

        system_prompt = AGENT_CONFIG['prompts']['system_prompt_step1'].format(
            current_datetime=get_current_date_info()
        )

        logging.info(f"Classifier params: {user_message}, {is_clarification}, {previous_summary}")

        if is_clarification and previous_summary is not None:
            system_prompt += f"""
            # ДОДАТКОВІ ІНСТРУКЦІЇ ДЛЯ УТОЧНЕННЯ:
            Це продовження діалогу. Використовуйте наданий нижче КОНТЕКСТ ДІАЛОГУ та ОСТАННЄ ПОВІДОМЛЕННЯ КОРИСТУВАЧА, щоб оновити ВСЮ JSON-структуру.
            
            Зокрема:
            - Оновіть "summary", об'єднавши стару та нову інформацію.
            - Якщо відповідь користувача містить достатньо даних для класифікації (наприклад, тепер відомо, холодна чи гаряча вода, або адресу знайдено), встановіть need_clarification: false. **НЕ** намагайтесь визначити фінальний код проблеми (H.1.1, U.4.3 тощо) — це завдання іншого кроку, просто оновіть summary та need_clarification.
            
            --- ПОЧАТОК КОНТЕКСТУ ---
            {previous_summary}
            --- КІНЕЦЬ КОНТЕКСТУ ---
            """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        logging.debug(f"Step 1: LLM request messages {json.dumps(messages, indent=2, ensure_ascii=False)}")

        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
            temperature=0.0,
            # max_tokens=200,
            response_format={"type": "json_object"}
        )

        step1_data = self.extract_content(response.choices[0].message.content)
        logging.info(f"Step 1 results \n {json.dumps(step1_data, indent=2, ensure_ascii=False)}")

        if step1_data.get('need_clarification', False):
            return {
                **step1_data
            }

        logging.info(f'Step 2 tags \n {json.dumps(step1_data.get("tags", []), indent=2, ensure_ascii=False)}')

        rag_results = categories.search_categories(step1_data.get("tags", []))
        if rag_results is None:
            logging.warning("FAISS не ініціалізовано або помилка пошуку, використовуємо порожній список")
            rag_results = []

        logging.info(f"Step 2 RAG results \n {json.dumps(rag_results, indent=2, ensure_ascii=False)}")

        formatted_prompt = AGENT_CONFIG['prompts']['system_prompt_step2'].format(
            current_datetime=get_current_date_info(),
            user_complaint=user_message,
            potential_categories_json=json.dumps(rag_results or []),
            chat_history=self.get_dialog(state.get('messages', [])),
            summary_description=step1_data.get('summary', {}).get("normalized_description"),
            summary_context_notes=step1_data.get('summary', {}).get("context_notes")
        )

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_message}
        ]

        logging.info(f"Step 2: LLM request messages {json.dumps(messages, indent=2, ensure_ascii=False)}")

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.0,
            # max_tokens=200,
            response_format={"type": "json_object"}
        )

        step2_data = self.extract_content(response.choices[0].message.content)

        print(json.dumps(step2_data, indent=2, ensure_ascii=False))
        logging.info(f"Step 2 results: {json.dumps(step2_data, indent=2, ensure_ascii=False)}")

        return {
            **step1_data,
            "problems": step2_data.get("problems", [])
        }

    @staticmethod
    def extract_content(content):
        content = get_content_as_str(content)
        result = get_content_as_json(content)
        logging.info(f"Extracted json {result}")
        return result

    def get_dialog(self, messages_history):
        formatted_history = []

        for message in messages_history:
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                formatted_history.append(f"Користувач: {content}")
            elif role == "assistant":
                # Можна додати повідомлення бота, наприклад, "Система: Уточніть адресу..."
                formatted_history.append(f"Система: {content}")

        # Об'єднуємо всі рядки історії в один великий блок
        return "\n".join(formatted_history)
