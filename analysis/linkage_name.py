from enum import Enum
from typing import List


class LinkageName(str, Enum):
    AVERAGE = "average"
    COMPLETE = "complete"
    SINGLE = "single"

    # Ward Linkage requires Euclidian distances, but we use cosine distances.

    @classmethod
    def get_names(cls) -> List[str]:
        return [name.value for name in cls]
