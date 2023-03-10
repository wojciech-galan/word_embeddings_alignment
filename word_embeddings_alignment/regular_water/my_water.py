import warnings
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import SupportsFloat, Union

from word_embeddings_alignment.SimpleAlignmentRepresentation import SimpleAlignmentRepresentation
from word_embeddings_alignment.my_warnings import MultipleEquallyScoredPathsFromMaxTo0
from word_embeddings_alignment.my_warnings import MultipleMaxValuesInDistanceMatrix
from word_embeddings_alignment.regular_water.matrices.EDNA_FULL import EDNAFULL_matrix

Numeric = Union[SupportsFloat, complex]
UPPER = 1
LEFT = 2
SLANT = 4
GAP_OPENED_FROM_UPPER = 8
GAP_OPENED_FROM_LEFT = 16
AMBIGUOUS_DIRECTIONS = {UPPER | LEFT, UPPER | SLANT, LEFT | SLANT}
GAP = '-'


def align(seq_a: str, seq_b: str, matrix: Dict[str, Dict[str, Numeric]], gap_open: Numeric,
          gap_extend: Numeric) -> SimpleAlignmentRepresentation:
	distance_matrix, traceback_matrix = create_distance_and_traceback_matrices(seq_a, seq_b, matrix, gap_open,
	                                                                           gap_extend)
	max_indices_list = find_indices_of_max(distance_matrix)
	if distance_matrix[max_indices_list[0]] == 0:
		alignment = SimpleAlignmentRepresentation(distance_matrix[max_indices_list[0]])
		return alignment
	elif len(max_indices_list) > 1:
		warnings.warn("Multiple best-scoring alignments are possible", MultipleMaxValuesInDistanceMatrix)
	return traceback(distance_matrix, max_indices_list[0], traceback_matrix, seq_a, seq_b)


def create_distance_and_traceback_matrices(seq_a: str, seq_b: str, matrix: Dict[str, Dict[str, Numeric]],
                                           gap_open: Numeric, gap_extend: Numeric) -> Tuple[np.ndarray, np.ndarray]:
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
			slant = distance_matrix[i - 1, j - 1] + matrix[char_a][char_b]
			upper = distance_matrix[i - 1, j] - (
				gap_extend if (
						i > 2 and j > 1 and traceback_matrix[i - 2, j - 1] & GAP_OPENED_FROM_UPPER) else gap_open)
			left = distance_matrix[i, j - 1] - (
				gap_extend if (i > 1 and j > 2 and traceback_matrix[i - 1, j - 2] & GAP_OPENED_FROM_LEFT) else gap_open)
			maximum = max(slant, upper, left, 0)
			distance_matrix[i, j] = maximum
			traceback_matrix[i - 1, j - 1] = (UPPER | GAP_OPENED_FROM_UPPER if maximum == upper else 0) | \
			                                 (LEFT | GAP_OPENED_FROM_LEFT if maximum == left else 0) | \
			                                 (SLANT if maximum == slant else 0)
	return distance_matrix, traceback_matrix


def traceback(distance_matrix: np.ndarray, max_element_indices: Tuple[int, int], traceback_matrix: np.ndarray,
              seq_a: str, seq_b: str):
	element = distance_matrix[max_element_indices]
	previously_used = 0
	curr_element_a, curr_element_b = max_element_indices
	alignment = SimpleAlignmentRepresentation(element)
	while element:
		direction = traceback_matrix[curr_element_a - 1, curr_element_b - 1]
		if direction & (SLANT | UPPER | LEFT) in AMBIGUOUS_DIRECTIONS:
			warnings.warn("Multiple best-scoring alignments are possible", MultipleEquallyScoredPathsFromMaxTo0)
		# first check, whether we are already extending a gap
		if (previously_used == LEFT) and (direction & LEFT):
			# gap on the first sequence
			alignment.add_data(GAP, seq_b[curr_element_b - 1])
			curr_element_b -= 1
			previously_used = LEFT
		elif (previously_used == UPPER) and (direction & UPPER):
			# gap on the second sequence
			alignment.add_data(seq_a[curr_element_a - 1], GAP)
			curr_element_a -= 1
			previously_used = UPPER
		# if we are not extending a gap:
		elif direction & SLANT:
			alignment.add_data(seq_a[curr_element_a - 1], seq_b[curr_element_b - 1])
			curr_element_a -= 1
			curr_element_b -= 1
			previously_used = 0
		elif direction & LEFT:
			# gap on the first sequence
			alignment.add_data(GAP, seq_b[curr_element_b - 1])
			curr_element_b -= 1
			previously_used = LEFT
		elif direction & UPPER:
			# gap on the second sequence
			alignment.add_data(seq_a[curr_element_a - 1], GAP)
			curr_element_a -= 1
			previously_used = UPPER
		element = distance_matrix[curr_element_a, curr_element_b]
	return alignment


def find_indices_of_max(array: np.ndarray) -> List[Tuple[int, int]]:
	maximum = array.max()
	indices = []
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i, j] == maximum:
				indices.append((i, j))
	return indices
