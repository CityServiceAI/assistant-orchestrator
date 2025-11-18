
class ServiceAgent:
    name = "service-agent"

    def run(self, issue_code):

        # На цьому етапі в нас вже має бути класифікована проблема її рівень, йле пошук в бд
        return {
            "name": 'Івано-Франківська філія ТОВ “Газорозподільні мережі України”',
            "contact-phone": '0 800 303 104',
            "emergency-phone": "104"
        }
