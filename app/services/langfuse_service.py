import logging
import os
from typing import Optional

try:
    from langfuse import Langfuse, observe

    try:
        from langfuse.langchain import CallbackHandler
    except Exception as e:
        logging.debug(f"Не вдалося імпортувати CallbackHandler: {e}")
        CallbackHandler = None
except Exception as e:
    logging.debug(f"Не вдалося імпортувати langfuse: {e}")
    Langfuse = None
    CallbackHandler = None
    observe = None


class LangfuseService:
    def __init__(self):
        self.enabled = False
        self.langfuse: Optional[Langfuse] = None
        self._init_langfuse()

    def _init_langfuse(self):
        if Langfuse is None:
            logging.debug(
                "langfuse не встановлено або не вдалося імпортувати. "
                "Встановіть: pip install langfuse"
            )
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logging.warning(
                "LANGFUSE_PUBLIC_KEY або LANGFUSE_SECRET_KEY не налаштовано. "
                "Langfuse вимкнено."
            )
            logging.debug(
                f"Перевірка env: LANGFUSE_PUBLIC_KEY={'встановлено' if public_key else 'відсутнє'}, "
                f"LANGFUSE_SECRET_KEY={'встановлено' if secret_key else 'відсутнє'}"
            )
            return

        try:
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            self.enabled = True
            logging.info(
                f"Langfuse увімкнено: host={host}, public_key={public_key[:10]}..."
            )
        except Exception as e:
            logging.error(f"Помилка ініціалізації Langfuse: {e}", exc_info=True)
            self.enabled = False

    def get_callback_handler(self, **kwargs) -> Optional[CallbackHandler]:
        if not self.enabled or CallbackHandler is None:
            return None

        try:
            handler = CallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                update_trace=kwargs.get("update_trace", True),
            )
            return handler
        except Exception as e:
            logging.error(f"Помилка створення CallbackHandler: {e}", exc_info=True)
            return None

    def create_trace(self, name: str, **kwargs):
        if not self.enabled or self.langfuse is None:
            return None

        try:
            user_id = kwargs.pop("user_id", None)
            session_id = kwargs.pop("session_id", None)
            span = self.langfuse.start_span(name=name, **kwargs)

            if user_id is not None or session_id is not None:
                span.update_trace(user_id=user_id, session_id=session_id)

            return span
        except Exception as e:
            logging.error(f"Помилка створення trace: {e}")
            return None

    def get_observe_decorator(self):
        if not self.enabled or observe is None:

            def no_op_decorator(*args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            return no_op_decorator
        return observe


_langfuse_service: Optional[LangfuseService] = None


def get_langfuse_service() -> LangfuseService:
    global _langfuse_service
    if _langfuse_service is None:
        _langfuse_service = LangfuseService()
    return _langfuse_service
