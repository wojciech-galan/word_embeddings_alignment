import warnings
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from word_embeddings_alignment.src.regular_water import regular_water
from word_embeddings_alignment.src.word_embeddings_water import word_embeddings_water
from word_embeddings_alignment.src.my_warnings import MultipleMaxValuesInDistanceMatrix
from word_embeddings_alignment.src.types import Numeric
from word_embeddings_alignment.src.simple_alignment_representation import SimpleAlignmentRepresentation

ALIGNMENT_TYPES = {
	'word_embeddings_water': word_embeddings_water,
	'regular_water': regular_water
}


def align(seq_a: str, seq_b: str, matrix: Dict[str, Numeric], gap_open: Numeric,
          gap_extend: Numeric, alignment_type: str) -> SimpleAlignmentRepresentation:
	create_distance_and_traceback_matrices = ALIGNMENT_TYPES[alignment_type].create_distance_and_traceback_matrices
	traceback = ALIGNMENT_TYPES[alignment_type].traceback
	distance_matrix, traceback_matrix = create_distance_and_traceback_matrices(seq_a, seq_b, matrix, gap_open,
	                                                                           gap_extend)
	max_indices_list = find_indices_of_max(distance_matrix)
	if distance_matrix[max_indices_list[0]] == 0:
		alignment = SimpleAlignmentRepresentation(distance_matrix[max_indices_list[0]])
		return alignment
	elif len(max_indices_list) > 1:
		warnings.warn("Multiple best-scoring alignments are possible", MultipleMaxValuesInDistanceMatrix)
	return traceback(distance_matrix, max_indices_list[0], traceback_matrix, seq_a, seq_b)


def find_indices_of_max(array: np.ndarray) -> List[Tuple[int, int]]:
	maximum = array.max()
	indices = []
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i, j] == maximum:
				indices.append((i, j))
	return indices
