from enum import Enum
from typing import List


class ModelName(str, Enum):
    CHARACTER_BERT = 'CharacterBERT'

    @classmethod
    def get_values(cls) -> List[str]:
        return list(map(lambda item: item.value, cls))
