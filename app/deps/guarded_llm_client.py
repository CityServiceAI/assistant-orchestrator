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
from app.services.langfuse_service import get_langfuse_service


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
        langfuse_service = get_langfuse_service()
        langfuse_span = None

        if langfuse_service.enabled and langfuse_service.langfuse:
            try:
                model_name = kwargs.get("model", "unknown")
                langfuse_span = langfuse_service.langfuse.start_generation(
                    name=f"llm_call_{model_name}",
                    model=model_name,
                    input=messages,
                    metadata={
                        "temperature": kwargs.get("temperature"),
                        "max_tokens": kwargs.get("max_tokens"),
                        "response_format": kwargs.get("response_format"),
                    },
                )
            except Exception as e:
                logging.debug(f"Не вдалося створити Langfuse span для LLM: {e}")

        input_check = self._check_user_input(messages)

        if input_check.is_blocked:
            error_msg = (
                f"Вхідний запит заблоковано Guardrails: {input_check.message}. "
                f"Тип порушення: {input_check.violation_type.value if input_check.violation_type else 'UNKNOWN'}"
            )
            logging.error(error_msg)
            if langfuse_span:
                try:
                    langfuse_span.update(
                        level="ERROR",
                        status_message=error_msg,
                        metadata={"guardrail_blocked": True},
                    )
                    langfuse_span.end()
                except Exception:
                    pass
            raise ValueError(error_msg)

        try:
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

            if langfuse_span:
                try:
                    output_content = (
                        response.choices[0].message.content
                        if response.choices
                        else None
                    )
                    usage_data = None
                    if response.usage:
                        usage_data = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                    langfuse_span.update(
                        output=output_content,
                        usage=usage_data,
                        metadata={
                            "guardrail_checked": True,
                            "response_id": response.id,
                        },
                    )
                    langfuse_span.end()
                except Exception as e:
                    logging.debug(f"Не вдалося оновити Langfuse span: {e}")

            return response
        except Exception as e:
            if langfuse_span:
                try:
                    langfuse_span.update(level="ERROR", status_message=str(e))
                    langfuse_span.end()
                except Exception:
                    pass
            raise


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
