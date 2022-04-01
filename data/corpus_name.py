from enum import Enum
from typing import List


class CorpusName(str, Enum):
    TOY = 'Toy'
    SEMCOR = 'SemCor'

    @classmethod
    def get_names(cls) -> List[str]:
        return [name.value for name in cls]
