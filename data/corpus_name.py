from enum import Enum
from typing import List


class CorpusName(str, Enum):
    SEMCOR = 'SemCor'
    SEMEVAL07 = "SemEval2007"
    SEMEVAL13 = "SemEval2013"
    SEMEVAL15 = "SemEval2015"
    SENSEVAL2 = "Senseval2"
    SENSEVAL3 = "Senseval3"
    TOY = 'Toy'

    @classmethod
    def get_values(cls) -> List[str]:
        return list(map(lambda item: item.value, cls))

    @property
    def is_wsdeval_name(self):
        """ Checks if the corpus is available in WSDEval. """
        return self in [self.SEMEVAL07, self.SEMEVAL13, self.SEMEVAL15,
                        self.SENSEVAL2, self.SENSEVAL3, self.SEMCOR]
