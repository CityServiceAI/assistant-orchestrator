from dataclasses import dataclass
from typing import List


@dataclass
class CategoryRecord:
    l1_code: str
    l1_name: str
    l2_code: str
    l2_name: str
    l3_code: str
    l3_name: str
    when_to_use: str
    diff_from_similar: str
    embedding: List[float] | None = None
