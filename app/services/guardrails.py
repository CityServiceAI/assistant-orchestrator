import logging
import os
from typing import Optional
from enum import Enum
import boto3
from botocore.exceptions import (
    ClientError,
    BotoCoreError,
    ReadTimeoutError,
    ConnectTimeoutError,
)


class GuardrailAction(str, Enum):
    GUARDRAIL_INTERVENED = "GUARDRAIL_INTERVENED"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def is_intervened(cls, action: str) -> bool:
        """Перевіряє, чи action означає блокування"""
        return "INTERVENED" in action.upper()


class GuardrailViolationType(str, Enum):
    CONTENT_FILTER = "CONTENT_FILTER"
    TOPIC = "TOPIC"
    WORD_FILTER = "WORD_FILTER"
    SENSITIVE_INFORMATION = "SENSITIVE_INFORMATION"
    CONTEXTUAL_GROUNDEDNESS = "CONTEXTUAL_GROUNDEDNESS"
    UNKNOWN = "UNKNOWN"


class GuardrailResult:
    def __init__(
        self,
        action: GuardrailAction,
        violation_type: Optional[GuardrailViolationType] = None,
        message: Optional[str] = None,
        filtered_content: Optional[str] = None,
    ):
        self.action = action
        self.violation_type = violation_type
        self.message = message
        self.filtered_content = filtered_content
        self.is_blocked = action == GuardrailAction.GUARDRAIL_INTERVENED

    def __repr__(self):
        return (
            f"GuardrailResult(action={self.action.value}, "
            f"violation_type={self.violation_type.value if self.violation_type else None}, "
            f"is_blocked={self.is_blocked})"
        )


class BedrockGuardrailsService:
    def __init__(
        self,
        guardrail_identifier: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        aws_region: Optional[str] = None,
    ):
        self.guardrail_identifier = guardrail_identifier or os.getenv(
            "AWS_BEDROCK_GUARDRAIL_ID"
        )
        raw_version = guardrail_version or os.getenv(
            "AWS_BEDROCK_GUARDRAIL_VERSION", "DRAFT"
        )
        self.guardrail_version = raw_version
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN")

        self.enabled = bool(self.guardrail_identifier) and boto3 is not None

        if not self.enabled:
            if not boto3:
                logging.warning(
                    "boto3 не встановлено. Guardrails вимкнено. "
                    "Встановіть: pip install boto3"
                )
            elif not self.guardrail_identifier:
                logging.warning(
                    "AWS_BEDROCK_GUARDRAIL_ID не налаштовано. Guardrails вимкнено."
                )
            return

        try:
            import botocore.config

            config = botocore.config.Config(
                read_timeout=10,
                connect_timeout=5,
                retries={"max_attempts": 1},
            )

            client_params = {
                "service_name": "bedrock-runtime",
                "region_name": self.aws_region,
                "config": config,
            }

            if self.aws_access_key_id and self.aws_secret_access_key:
                client_params.update(
                    {
                        "aws_access_key_id": self.aws_access_key_id,
                        "aws_secret_access_key": self.aws_secret_access_key,
                    }
                )
                if self.aws_session_token:
                    client_params["aws_session_token"] = self.aws_session_token

            try:
                self.bedrock_runtime = boto3.client(**client_params)
                logging.info(
                    f"Bedrock Guardrails увімкнено: guardrail_id={self.guardrail_identifier}, "
                    f"version={self.guardrail_version}, region={self.aws_region}"
                )
            except Exception as client_error:
                logging.error(f"Помилка створення boto3 клієнта: {client_error}")
                raise
        except Exception as e:
            logging.error(
                f"Помилка ініціалізації Bedrock Guardrails: {e}", exc_info=True
            )
            self.enabled = False

    def _create_runtime_client(self):
        import botocore.config

        config = botocore.config.Config(
            read_timeout=10,
            connect_timeout=5,
            retries={"max_attempts": 1},
        )

        client_params = {
            "service_name": "bedrock-runtime",
            "region_name": self.aws_region,
            "config": config,
        }

        if self.aws_access_key_id and self.aws_secret_access_key:
            client_params.update(
                {
                    "aws_access_key_id": self.aws_access_key_id,
                    "aws_secret_access_key": self.aws_secret_access_key,
                }
            )
            if self.aws_session_token:
                client_params["aws_session_token"] = self.aws_session_token

        try:
            client = boto3.client(**client_params)
            return client
        except Exception as e:
            logging.error(f"Помилка створення boto3 клієнта: {e}")
            raise

    def check_content(
        self,
        text: str,
        content_type: str = "INPUT",
    ) -> GuardrailResult:
        if not self.enabled or not text:
            return GuardrailResult(GuardrailAction.NONE)

        try:
            if not hasattr(self, "bedrock_runtime") or not self.bedrock_runtime:
                logging.warning("bedrock_runtime не існує, створюємо новий клієнт")
                runtime_client = self._create_runtime_client()
            else:
                runtime_client = self.bedrock_runtime

            response = runtime_client.apply_guardrail(
                guardrailIdentifier=self.guardrail_identifier,
                guardrailVersion=self.guardrail_version,
                source=content_type,
                content=[
                    {
                        "text": {
                            "text": text,
                        }
                    }
                ],
            )

            action = response.get("action", GuardrailAction.UNKNOWN.value)

            if GuardrailAction.is_intervened(action):
                violations = response.get("violations", [])
                violation_type = GuardrailViolationType.UNKNOWN
                message_parts = []

                for violation in violations:
                    v_type = violation.get("type", "UNKNOWN")
                    v_message = violation.get("message", "")

                    try:
                        violation_type = GuardrailViolationType(v_type)
                    except ValueError:
                        violation_type = GuardrailViolationType.UNKNOWN

                    message_parts.append(v_message)

                message = (
                    "; ".join(message_parts)
                    if message_parts
                    else "Контент заблоковано Guardrail"
                )

                filtered_content = None
                content_data = response.get("content")
                if content_data:
                    if isinstance(content_data, list) and len(content_data) > 0:
                        text_obj = content_data[0].get("text", {})
                        if isinstance(text_obj, dict):
                            filtered_content = text_obj.get("text")
                    elif isinstance(content_data, dict):
                        text_obj = content_data.get("text", {})
                        if isinstance(text_obj, dict):
                            filtered_content = text_obj.get("text")
                        else:
                            filtered_content = text_obj

                logging.warning(
                    f"Guardrail заблокував {content_type}: {message} "
                    f"(violation_type={violation_type.value})"
                )

                return GuardrailResult(
                    action=GuardrailAction.GUARDRAIL_INTERVENED,
                    violation_type=violation_type,
                    message=message,
                    filtered_content=filtered_content,
                )

            return GuardrailResult(GuardrailAction.NONE)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            logging.error(
                f"Помилка AWS Bedrock Guardrails API ({error_code}): {error_message}"
            )

            if error_code == "UnrecognizedClientException":
                logging.warning(
                    "Guardrails недоступний через проблеми з credentials. Продовжуємо без перевірки guardrails."
                )
                return GuardrailResult(GuardrailAction.NONE)

            if error_code == "ValidationException":
                logging.warning(
                    "Guardrails недоступний через ValidationException. Продовжуємо без перевірки guardrails."
                )
                return GuardrailResult(GuardrailAction.NONE)

            logging.warning(
                f"Guardrails API помилка ({error_code}). Продовжуємо без перевірки guardrails."
            )
            return GuardrailResult(GuardrailAction.NONE)

        except (ReadTimeoutError, ConnectTimeoutError) as e:
            logging.warning(f"Таймаут Guardrails API: {e}")
            return GuardrailResult(
                GuardrailAction.UNKNOWN,
                message=f"Таймаут Guardrails API: {str(e)}",
            )
        except BotoCoreError as e:
            logging.error(f"Помилка boto3: {e}")
            return GuardrailResult(
                GuardrailAction.UNKNOWN,
                message=f"Boto3 помилка: {str(e)}",
            )
        except Exception as e:
            logging.error(f"Помилка Guardrails: {e}", exc_info=True)
            return GuardrailResult(
                GuardrailAction.UNKNOWN,
                message=f"Неочікувана помилка: {str(e)}",
            )

    def check_user_input(self, text: str) -> GuardrailResult:
        return self.check_content(text, content_type="INPUT")

    def check_model_response(self, text: str) -> GuardrailResult:
        return self.check_content(text, content_type="OUTPUT")


_guardrails_service: Optional[BedrockGuardrailsService] = None


def get_guardrails_service() -> BedrockGuardrailsService:
    global _guardrails_service
    if _guardrails_service is None:
        _guardrails_service = BedrockGuardrailsService()
    return _guardrails_service
