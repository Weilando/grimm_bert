from enum import Enum
from typing import List


class CorpusName(str, Enum):
    TOY = 'toy'
    SEMCOR = 'semcor'
    SEMEVAL07 = "semeval2007"
    SEMEVAL13 = "semeval2013"
    SEMEVAL15 = "semeval2015"
    SENSEVAL2 = "senseval2"
    SENSEVAL3 = "senseval3"

    @classmethod
    def get_names(cls) -> List[str]:
        return list(map(lambda name: name.value, cls))

    @property
    def is_wsdeval_name(self):
        """ Checks if the corpus is available in WSDEval. """
        return self in [self.SEMEVAL07, self.SEMEVAL13, self.SEMEVAL15,
                        self.SENSEVAL2, self.SENSEVAL3]
