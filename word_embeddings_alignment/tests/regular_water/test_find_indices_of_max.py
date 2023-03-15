import numpy as np
import pytest
from word_embeddings_alignment.src.regular_water.regular_water import find_indices_of_max


@pytest.fixture(scope='function')
def one_max_array() -> np.ndarray:
	return np.array([[1, 2], [4, 3]])


@pytest.fixture(scope='function')
def multiple_max_array() -> np.ndarray:
	return np.array([[4, 2], [4, 3]])


def test_one_max(one_max_array: np.ndarray):
	assert find_indices_of_max(one_max_array) == [(1, 0)]


def test_multiple_max(multiple_max_array: np.ndarray):
	assert find_indices_of_max(multiple_max_array) == [(0, 0), (1, 0)]
