from typing import Tuple
from typing import Dict
from typing import SupportsFloat, Union

import numpy as np

Numeric = Union[SupportsFloat, complex]
UPPER = 1
LEFT = 2
SLANT = 4
GAP_OPENED_FROM_UPPER = 8
GAP_OPENED_FROM_LEFT = 16
AMBIGUOUS_DIRECTIONS = {UPPER | LEFT, UPPER | SLANT, LEFT | SLANT}
GAP = '-'


def create_distance_and_traceback_matrices(seq_a: str, seq_b: str, word_embeddings: Dict[str, np.ndarray],
                                           gap_open: Numeric, gap_extend: Numeric) -> Tuple[np.ndarray, np.ndarray]:
	word_length = get_first_key_from_a_dict(word_embeddings)
	# create initial matrix
	distance_matrix = np.full((len(seq_a) + 1, len(seq_b) + 1), np.NaN)
	traceback_matrix = np.zeros((len(seq_a), len(seq_b)), dtype=np.byte)
	# initialize first row and column
	distance_matrix[0, :] = 0
	distance_matrix[:, 0] = 0
	# fill the matrix
	for i, char_a in enumerate(seq_a, 1):
		for j, char_b in enumerate(seq_b, 1):
			# gap_penalty equals gap_extend if there is already a gap in a previous cell, else gap_open
			if (i + 3 <= len(seq_a)) and (j + 3 <= len(seq_b)):
				to_subtract = min(max(i, j), 3)
				slant = distance_matrix[i - to_subtract, j - to_subtract] + euclidean_distance(word_embeddings[seq_a[i:i+3]], word_embeddings[seq_b[j:j+3]])
			else:
				slant = np.inf
			upper = distance_matrix[i - 1, j] - (
				gap_extend if (
						i > 2 and j > 1 and traceback_matrix[i - 2, j - 1] & GAP_OPENED_FROM_UPPER) else gap_open)
			left = distance_matrix[i, j - 1] - (
				gap_extend if (i > 1 and j > 2 and traceback_matrix[i - 1, j - 2] & GAP_OPENED_FROM_LEFT) else gap_open)
			minimum = min(slant, upper, left, np.inf)
			distance_matrix[i, j] = minimum
			traceback_matrix[i - 1, j - 1] |= (UPPER | GAP_OPENED_FROM_UPPER if minimum == upper else 0) | \
			                                 (LEFT | GAP_OPENED_FROM_LEFT if minimum == left else 0) | \
			                                 (SLANT if minimum == slant else 0)
	return distance_matrix, traceback_matrix


def get_first_key_from_a_dict(a_dict: Dict):
	return next(iter(a_dict))


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray):
	return np.linalg.norm(vector1-vector2)


print(euclidean_distance(np.array([3, 2]), np.array([4, 1])))
