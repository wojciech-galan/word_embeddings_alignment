import pytest
from typing import Dict
import numpy as np

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


@pytest.fixture(scope="session")
def linear_gap__distance_matrix_1() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[0, 5, 5, 0, 5, 0, 5],
		[0, 5, 10, 5, 5, 1, 5],
		[0, 5, 10, 6, 10, 5, 6],
		[0, 5, 10, 6, 11, 6, 10],
		[0, 5, 10, 6, 11, 7, 11],
		[0, 5, 10, 6, 11, 7, 12]
	])


@pytest.fixture(scope="session")
def linear_gap__traceback_matrix_1() -> np.ndarray:
	return np.array([
		[4, 4, 18, 4, 18, 4],
		[4, 4, 18, 4, 4, 4],
		[4, 4, 4, 4, 18, 4],
		[4, 4, 4, 4, 22, 4],
		[4, 4, 4, 4, 4, 4],
		[4, 4, 4, 4, 4, 4]
	], dtype=np.byte)


@pytest.fixture(scope="session")
def linear_gap__distance_matrix_2() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 5, 5, 4, 5, 4, 5, 5],
		[0, 5, 10, 9, 9, 8, 9, 10],
		[0, 5, 10, 9, 14, 13, 13, 14],
		[0, 4, 9, 15, 14, 19, 18, 17],
		[0, 5, 9, 14, 20, 19, 24, 23],
		[0, 5, 10, 13, 19, 18, 24, 29],
		[0, 5, 10, 12, 18, 17, 23, 29]
	])


@pytest.fixture(scope="session")
def linear_gap__traceback_matrix_2() -> np.ndarray:
	return np.array([
		[4, 4, 18, 4, 18, 4, 4],
		[4, 4, 18, 4, 18, 4, 4],
		[4, 4, 18, 4, 18, 4, 4],
		[9, 9, 4, 18, 4, 18, 18],
		[4, 4, 9, 4, 18, 4, 22],
		[4, 4, 9, 13, 27, 4, 4],
		[4, 4, 9, 13, 27, 13, 4]
	], dtype=np.byte)


@pytest.fixture(scope="session")
def linear_gap__distance_matrix_3() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 5, 4, 5, 5, 4, 3, 2, 1],
		[0, 4, 10, 9, 8, 10, 9, 8, 7],
		[0, 5, 9, 15, 14, 13, 12, 11, 10],
		[0, 4, 8, 14, 13, 12, 11, 17, 16],
		[0, 3, 7, 13, 12, 11, 17, 16, 22]
	])


@pytest.fixture(scope="session")
def linear_gap__traceback_matrix_3() -> np.ndarray:
	return np.array([
		[4, 18, 4, 4, 18, 18, 18, 18],
		[9, 4, 18, 18, 4, 18, 18, 18],
		[4, 9, 4, 22, 18, 18, 18, 18],
		[9, 9, 9, 27, 27, 27, 4, 18],
		[9, 9, 9, 27, 27, 4, 27, 4]
	], dtype=np.byte)


@pytest.fixture(scope="session")
def affine_gap__distance_matrix_1() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 5, 0, 5, 5, 0, 0, 0, 0],
		[0, 0, 10, 5, 4, 10, 5, 4, 3],
		[0, 5, 5, 15, 10, 9, 8, 7, 6],
		[0, 0, 4, 10, 11, 6, 5, 13, 8],
		[0, 0, 3, 9, 6, 7, 11, 8, 18]
	])


@pytest.fixture(scope="session")
def affine_gap__traceback_matrix_1() -> np.ndarray:
	return np.array([
		[4, 18, 4, 4, 18, 0, 0, 0],
		[9, 4, 18, 18, 4, 18, 18, 18],
		[4, 9, 4, 22, 18, 18, 18, 18],
		[9, 9, 9, 4, 22, 22, 4, 18],
		[0, 9, 9, 13, 4, 4, 9, 4]
	], dtype=np.ndarray)
