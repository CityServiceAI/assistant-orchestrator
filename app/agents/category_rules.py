from typing import Optional


class PreClassificationResult(dict):
    """
    Простий контейнер для результату правил:
    {
      "category": str,                # L3 Код (наприклад, "E.4.1" або "Z.1.1")
      "confidence": float,            # 0..1
      "need_clarification": bool,
      "clarification_question": str | None
    }
    """

    pass


def pre_classification_rules(text: str) -> Optional[PreClassificationResult]:
    """
    Швидко відсікаємо очевидні кейси:
    - аварійні служби
    - явно не комунальна тема
    Якщо можемо щось визначити — повертаємо PreClassificationResult, інакше None.
    """
    t = text.lower()

    # --- 1. Аварійна газова ---
    if any(p in t for p in ["запах газу", "пахне газом", "витік газу", "витікає газ"]):
        return PreClassificationResult(
            category="E.4.1",
            confidence=1.0,
            need_clarification=False,
            clarification_question=None,
        )

    # --- 2. Пожежа / МНС ---
    if any(p in t for p in ["пожежа", "горить будинок", "щось горить", "загорілося"]):
        return PreClassificationResult(
            category="E.1.1",
            confidence=1.0,
            need_clarification=False,
            clarification_question=None,
        )

    # --- 3. Явно не комунальна сфера (NOT_MUNICIPAL, диспетчер пояснить) ---
    not_municipal_markers = [
        "інтернет",
        "wi-fi",
        "вайфай",
        "провайдер",
        "мобільний оператор",
        "lifecell",
        "київстар",
        "vodafone",
        "банкомат",
        "приват24",
        "монобанк",
        "банк",
        "кредитна карта",
        "картка",
        "доставка",
        "курʼєр",
        "курьер",
        "glovo",
        "bolt",
        "ubereats",
        "rocket",
    ]
    if any(w in t for w in not_municipal_markers):
        return PreClassificationResult(
            category="Z.1.1",
            confidence=0.95,
            need_clarification=False,
            clarification_question=None,
        )

    # TODO: сюди можна додати ще прості правила для типових кейсів,
    # наприклад, явний "ліфт" → код ліфта, "сміття" → код сміття і т.д.

    return None
