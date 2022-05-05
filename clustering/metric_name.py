from enum import Enum
from typing import List


class MetricName(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

    @classmethod
    def get_names(cls) -> List[str]:
        return list(map(lambda name: name.value, cls))
