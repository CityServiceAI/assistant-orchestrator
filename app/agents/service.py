import logging
import json
import os
import app.data.contacts_faiss as contacts
from app.tools.response import assistant_msg, load_config, get_content_as_str, get_content_as_json
from app.deps.litellm_client import client

CONFIG_FILE = os.getenv("SERVICE_YAML", "app/agents/service.yaml")
AGENT_CONFIG = load_config(CONFIG_FILE)


class ServiceAgent:
    name = "service-agent"

    @staticmethod
    def format_location_for_rag(location_details):
        """
        Перетворює словник location_details на стандартизований рядок для RAG-пошуку.

        Args:
            location_details (dict): Словник з деталями адреси
                                    (наприклад, {'city': 'Київ', 'street': 'Грекова', 'building_number': '3', 'apartment': None}).

        Returns:
            str: Відформатований рядок адреси.
        """
        if not location_details or location_details.get("city") is None:
            return "" # Повертаємо порожній рядок, якщо даних про локацію немає

        city = location_details.get("city", "")
        street = location_details.get("street", "")
        building = location_details.get("building_number", "")
        apartment = location_details.get("apartment", "")

        address_string = f"{city}, вул. {street}, буд. {building}"

        if apartment:
            address_string += f", кв. {apartment}"

        return address_string.strip() # Прибираємо зайві пробіли

    def run(self, state):

        summary = state.get("summary")
        problems = state.get("problems", [])
        location_type = state.get("location_type")
        location_details = state.get("location_details")


        # query_for_contact_rag = f'{summary["normalized_description"]} {self.format_location_for_rag(location_details)}'
        # search_1 = contacts.search(query_for_contact_rag)
        # logging.info(f'Пошук по summary \n {summary["normalized_description"]} \n \n {summary["context_notes"]} \n {json.dumps(search_1, indent=2, ensure_ascii=False)}')

        organisations = []

        for problem in problems:
            # Використовуємо category_name та location_type для пошуку КОНКРЕТНОГО контакту
            # query_contact = f"{problem['category_name']} {location_type} {problem['responsible_entity_type']}"
            responsible_entity_type = problem['responsible_entity_type']
            query_contact = f"{summary['normalized_description']} {location_type} {self.format_location_for_rag(location_details)} {responsible_entity_type}"
            search_2 = contacts.search(query_contact)

            if responsible_entity_type == "ManagementCompany/OSBB":
                fallback = contacts.get_fallback_by_city(location_details.get("city"))
                if fallback:
                    search_2.append(fallback)

            logging.info(f'Пошук по проблемі {problem}')
            logging.info(f'Пошук по проблемі {json.dumps(search_2, indent=2, ensure_ascii=False)}')
            organisations.extend(search_2)

        logging.debug(f'Знайдені організації {json.dumps(organisations, indent=2, ensure_ascii=False)}')

        if len(organisations) < 1:
            return {
                "is_no_service": True
            }

        logging.info(f"Organisations: {self.format_organisations_for_prompt(organisations)}")

        system_prompt = AGENT_CONFIG["prompts"]["select-org-prompt"].format(
            context=json.dumps(state),
            contacts=self.format_organisations_for_prompt(organisations)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Проаналізуйте інформацію та виберіть контакти."}
        ]

        response = client.chat.completions.create(
            model='gpt-4.1',
            messages=messages,
            temperature=0.0,
            # max_tokens=200,
            response_format={"type": "json_object"}
        )

        json_data = self.extract_content(response.choices[0].message.content)
        logging.info(f"Serch service:Extracted json \n {json.dumps(json_data, indent=2, ensure_ascii=False)}")
        return {
            **json_data,
            "usage": response.usage.to_dict(),
        }

    @staticmethod
    def extract_content(content):
        content = get_content_as_str(content)
        return get_content_as_json(content)

    @staticmethod
    def format_organisation(org):
        return f'Відповідальна служба: {org}'

    def format_organisations_for_prompt(self, organisations):
        result = list(map(self.org_to_str, organisations))
        return "\n".join(result)

    def org_to_str(self, org):
        result = []
        for key, value in org.items():
            result.append(f"{key}: {value}")

        return " | ".join(result)



