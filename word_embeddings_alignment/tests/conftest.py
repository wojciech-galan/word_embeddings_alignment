import pytest
from typing import Dict
import numpy as np

# EDNAFULL_SIMPLIFIED
#     A   T   G   C
# A   5  -4  -4  -4
# T  -4   5  -4  -4
# G  -4  -4   5  -4
# C  -4  -4  -4   5

EDNAFULL_SIMPLIFIED = {'AA': 5,
                       'AT': -4,
                       'AG': -4,
                       'AC': -4,
                       'TA': -4,
                       'TT': 5,
                       'TG': -4,
                       'TC': -4,
                       'GA': -4,
                       'GT': -4,
                       'GG': 5,
                       'GC': -4,
                       'CA': -4,
                       'CT': -4,
                       'CG': -4,
                       'CC': 5,
                       }


@pytest.fixture(scope="package")
def ednafull_simplified() -> Dict[str, Dict[str, int]]:
	return EDNAFULL_SIMPLIFIED


@pytest.fixture(scope="package")
def embeddings() -> Dict[str, np.ndarray]:
	return {
		'ACG': np.array([0, 0]),
		'GTA': np.array([0, 1]),
		'TAC': np.array([0, 2]),
		'CGA': np.array([0, 3]),
		'CGT': np.array([0, 4]),
		'GAC': np.array([0, 5]),
		'GTT': np.array([0, 7]),
		'TTA': np.array([0, 9])
	}