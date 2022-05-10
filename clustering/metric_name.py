from enum import Enum
from typing import List


class MetricName(str, Enum):
    COSINE = "Cosine"
    EUCLIDEAN = "Euclidean"

    @classmethod
    def get_values(cls) -> List[str]:
        return list(map(lambda item: item.value, cls))
