import pytest
from typing import Dict

# EDNAFULL_SIMPLIFIED
#     A   T   G   C
# A   5  -4  -4  -4
# T  -4   5  -4  -4
# G  -4  -4   5  -4
# C  -4  -4  -4   5


EDNAFULL_SIMPLIFIED = {'A': {'A': 5,
                             'T': -4,
                             'G': -4,
                             'C': -4},
                       'T': {'A': -4,
                             'T': 5,
                             'G': -4,
                             'C': -4},
                       'G': {'A': -4,
                             'T': -4,
                             'G': 5,
                             'C': -4},
                       'C': {'A': -4,
                             'T': -4,
                             'G': -4,
                             'C': 5}
                       }


@pytest.fixture(scope="session")
def ednafull_simplified() -> Dict[str, Dict[str, int]]:
    return EDNAFULL_SIMPLIFIED
