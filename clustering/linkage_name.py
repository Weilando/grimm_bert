from enum import Enum
from typing import List


class LinkageName(str, Enum):
    AVERAGE = "Average"
    COMPLETE = "Complete"
    SINGLE = "Single"

    # Ward Linkage requires Euclidean distances.

    @classmethod
    def get_values(cls) -> List[str]:
        return list(map(lambda item: item.value, cls))
