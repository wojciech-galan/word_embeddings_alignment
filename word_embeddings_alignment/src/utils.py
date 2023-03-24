import re
import warnings
from typing import Dict
from typing import List
from typing import Tuple
from typing import Generator

import numpy as np

from word_embeddings_alignment.src.regular_water import regular_water
from word_embeddings_alignment.src.word_embeddings_water import word_embeddings_water
from word_embeddings_alignment.src.my_warnings import MultipleMaxValuesInDistanceMatrix
from word_embeddings_alignment.src.types import Numeric
from word_embeddings_alignment.src.simple_alignment_representation import SimpleAlignmentRepresentation

FASTA_PATTERN = re.compile('^>(\S+)\s+([^>]+)', re.MULTILINE)
ALIGNMENT_TYPES = {
	'word_embeddings': word_embeddings_water,
	'classic': regular_water
}


def align(seq_a: str, seq_b: str, matrix: Dict[str, Numeric], gap_open: Numeric,
          gap_extend: Numeric, alignment_type: str,
          return_multiple_alignments: bool = False) -> Generator[SimpleAlignmentRepresentation, None, None]:
	create_distance_and_traceback_matrices = ALIGNMENT_TYPES[alignment_type].create_distance_and_traceback_matrices
	traceback = ALIGNMENT_TYPES[alignment_type].traceback
	distance_matrix, traceback_matrix = create_distance_and_traceback_matrices(seq_a, seq_b, matrix, gap_open,
	                                                                           gap_extend)
	max_indices_list = find_indices_of_max(distance_matrix)
	if distance_matrix[max_indices_list[0]] == 0:
		alignment = SimpleAlignmentRepresentation(distance_matrix[max_indices_list[0]], max_indices_list[0][0] - 1, max_indices_list[0][1] - 1)
		yield alignment
	elif return_multiple_alignments:
		for max_indices in max_indices_list:
			yield traceback(distance_matrix, max_indices, traceback_matrix, seq_a, seq_b)
	else:
		if len(max_indices_list) > 1:
			warnings.warn("Multiple best-scoring alignments are possible", MultipleMaxValuesInDistanceMatrix)
		yield traceback(distance_matrix, max_indices_list[0], traceback_matrix, seq_a, seq_b)


def find_indices_of_max(array: np.ndarray) -> List[Tuple[int, int]]:
	maximum = array.max()
	indices = []
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i, j] == maximum:
				indices.append((i, j))
	return indices


def read_raw_seq(f_path: str) -> str:
	with open(f_path) as f:
		return ''.join(f.readlines())


def read_fasta_seq(f_path: str) -> str:
	with open(f_path) as f:
		for id_, seq in re.findall(FASTA_PATTERN, f.read()):
			yield id_, ''.join(seq.split())


def write_fasta_alignment():
	pass