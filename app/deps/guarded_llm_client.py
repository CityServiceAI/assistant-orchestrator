import logging
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion

from app.services.guardrails import (
    get_guardrails_service,
    GuardrailResult,
    GuardrailAction,
)
from app.deps.litellm_client import _base_client as base_client


class GuardedCompletions:
    def __init__(self, base_completions, guardrails_service):
        self.base_completions = base_completions
        self.guardrails = guardrails_service

    def _extract_user_content(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        user_messages = [
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user" and msg.get("content")
        ]
        return " ".join(user_messages) if user_messages else None

    def _check_user_input(self, messages: List[Dict[str, Any]]) -> GuardrailResult:
        user_content = self._extract_user_content(messages)
        if not user_content:
            return GuardrailResult(GuardrailAction.NONE)

        result = self.guardrails.check_user_input(user_content)

        if result.is_blocked:
            logging.warning(
                f"Guardrail заблокував вхідний запит користувача: {result.message}"
            )

        return result

    def _check_model_response(self, content: str) -> GuardrailResult:
        if not content:
            return GuardrailResult(GuardrailAction.NONE)

        result = self.guardrails.check_model_response(content)

        if result.is_blocked:
            logging.warning(f"Guardrail заблокував відповідь моделі: {result.message}")
        elif result.violation_type and "GROUNDEDNESS" in result.violation_type.value:
            logging.warning(
                f"Guardrail виявив можливі галюцинації у відповіді моделі: {result.message}"
            )

        return result

    def create(self, messages: List[Dict[str, Any]], **kwargs) -> ChatCompletion:
        input_check = self._check_user_input(messages)

        if input_check.is_blocked:
            error_msg = (
                f"Вхідний запит заблоковано Guardrails: {input_check.message}. "
                f"Тип порушення: {input_check.violation_type.value if input_check.violation_type else 'UNKNOWN'}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        response = self.base_completions.create(messages=messages, **kwargs)

        if response.choices and len(response.choices) > 0:
            model_content = response.choices[0].message.content

            if model_content:
                response_check = self._check_model_response(model_content)

                if response_check.is_blocked:
                    logging.error(
                        f"Відповідь моделі заблокована Guardrails: {response_check.message}"
                    )
                    response.choices[0].message.content = (
                        response_check.filtered_content
                        or "Вибачте, я не можу надати відповідь на це питання через політики безпеки."
                    )
                elif (
                    response_check.violation_type
                    and "GROUNDEDNESS" in response_check.violation_type.value
                ):
                    logging.warning(
                        f"Можливі галюцинації у відповіді: {response_check.message}"
                    )

        return response


class GuardedChat:
    def __init__(self, base_chat, guardrails_service):
        self.base_chat = base_chat
        self.guardrails = guardrails_service
        self._completions = GuardedCompletions(
            base_chat.completions, guardrails_service
        )

    @property
    def completions(self):
        return self._completions


class GuardedLLMClient:
    def __init__(self, base_client: AzureOpenAI):
        self.base_client = base_client
        self.guardrails = get_guardrails_service()
        self._chat = GuardedChat(base_client.chat, self.guardrails)

    @property
    def chat(self):
        return self._chat

    def __getattr__(self, name):
        return getattr(self.base_client, name)


guarded_client = GuardedLLMClient(base_client)
client = guarded_client
