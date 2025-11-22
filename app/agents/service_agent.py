import logging
import json
import app.data.contacts_faiss as contacts
from app.tools.response import assistant_msg


class ServiceAgent:
    name = "service-agent"

    def format_location_for_rag(self, location_details):
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

    def run(self, summary, problems, location_type, location_details):

        # query_for_contact_rag = f'{summary["normalized_description"]} {self.format_location_for_rag(location_details)}'
        # search_1 = contacts.search(query_for_contact_rag)
        # logging.info(f'Пошук по summary \n {summary["normalized_description"]} \n \n {summary["context_notes"]} \n {json.dumps(search_1, indent=2, ensure_ascii=False)}')

        messages = []

        for problem in problems:
            # Використовуємо category_name та location_type для пошуку КОНКРЕТНОГО контакту
            # query_contact = f"{problem['category_name']} {location_type} {problem['responsible_entity_type']}"
            query_contact = f"{summary['normalized_description']} {location_type} {self.format_location_for_rag(location_details)} {problem['responsible_entity_type']}"
            search_2 = contacts.search(query_contact, 1)
            for org in search_2:
                messages.append(assistant_msg(self.format_organisation(org)))
            logging.debug(f'Пошук по проблемі {problem}')
            logging.debug(f'Пошук по проблемі {json.dumps(search_2, indent=2, ensure_ascii=False)}')



        # На цьому етапі в нас вже має бути класифікована проблема її рівень, йле пошук в бд
        return {
            "messages": messages
        }

    def format_organisation(self, org):
        return f'Відповідальна служба: {org}'
