from enum import Enum
from typing import List


class LinkageName(str, Enum):
    AVERAGE = "average"
    COMPLETE = "complete"
    SINGLE = "single"

    # Ward Linkage requires Euclidean distances.

    @classmethod
    def get_names(cls) -> List[str]:
        return list(map(lambda name: name.value, cls))
