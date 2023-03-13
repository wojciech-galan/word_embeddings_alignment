import warnings
from typing import Tuple
from typing import Dict
from typing import SupportsFloat, Union
from word_embeddings_alignment.simple_alignment_representation import SimpleAlignmentRepresentation
from word_embeddings_alignment.my_warnings import MultipleEquallyScoredPathsFromMaxTo0
from word_embeddings_alignment.my_warnings import MultipleMaxValuesInDistanceMatrix

import numpy as np

Numeric = Union[SupportsFloat, complex]
UPPER = 1
LEFT = 2
SLANT = 4
AMBIGUOUS_DIRECTIONS = {UPPER | LEFT, UPPER | SLANT, LEFT | SLANT}
GAP = '-'


def create_distance_and_traceback_matrices(seq_a: str, seq_b: str, word_embeddings: Dict[str, np.ndarray],
                                           gap_open: Numeric, gap_extend: Numeric) -> Tuple[np.ndarray, np.ndarray]:
	word_length = len(get_first_key_from_a_dict(word_embeddings))
	if word_length != 3:
		raise NotImplementedError
	# create initial matrix
	distance_matrix = np.zeros((len(seq_a) + 1, len(seq_b) + 1))
	traceback_matrix = np.zeros((len(seq_a), len(seq_b)), dtype=np.byte)
	# fill the matrix
	for i, char_a in enumerate(seq_a, 1):
		for j, char_b in enumerate(seq_b, 1):
			# gap_penalty equals gap_extend if there is already a gap in a previous cell, else gap_open
			if (i + 2 <= len(seq_a)) and (j + 2 <= len(seq_b)):
				slant = distance_matrix[i - 1, j - 1] + points_for_word_embeddings(word_embeddings, seq_a[i-1:i+2], seq_b[j-1:j+2])
			else:
				slant = 0
			upper = distance_matrix[i - 1, j] - (
				gap_extend if (
						i > 2 and j > 1 and traceback_matrix[i - 2, j - 1] & UPPER) else gap_open)
			left = distance_matrix[i, j - 1] - (
				gap_extend if (i > 1 and j > 2 and traceback_matrix[i - 1, j - 2] & LEFT) else gap_open)
			maximum = max(slant, upper, left, 0)
			print(i, j, slant, upper, left)
			# import pdb
			# if i == 4 and j == 4:
			# 	pdb.set_trace()
			if maximum == 0:
				pass
			elif maximum == slant:
				# update of +word_length position in matrices
				traceback_matrix[i + 1, j + 1] |= SLANT
				distance_matrix[i+2, j+2] = slant
				# update of current position in matrices
				curr_max = max(upper, left, distance_matrix[i, j])
				if curr_max:
					# first clear current value of traceback_matrix[i, j]
					traceback_matrix[i - 1, j - 1] = 0
					# then update it
					traceback_matrix[i - 1, j - 1] |= SLANT if curr_max == distance_matrix[i, j] else 0
					traceback_matrix[i - 1, j - 1] |= UPPER if curr_max == upper else 0
					traceback_matrix[i - 1, j - 1] |= LEFT if curr_max == left else 0
					distance_matrix[i, j] = curr_max
			# in the code below: maximum == upper or maximum == left
			elif traceback_matrix[i - 1, j - 1] == SLANT and distance_matrix[i, j] > maximum:
				pass
			elif traceback_matrix[i - 1, j - 1] == SLANT and distance_matrix[i, j] == maximum:
				traceback_matrix[i - 1, j - 1] |= UPPER if maximum == upper else 0
				traceback_matrix[i - 1, j - 1] |= LEFT if maximum == left else 0
			elif (traceback_matrix[i - 1, j - 1] == SLANT and distance_matrix[i, j] < maximum) \
					or \
				 (traceback_matrix[i - 1, j - 1] == 0):
				distance_matrix[i, j] = maximum
				traceback_matrix[i - 1, j - 1] = UPPER if maximum == upper else 0
				traceback_matrix[i - 1, j - 1] |= LEFT if maximum == left else 0
			else:
				# should never happen
				pdb.set_trace()

			print(distance_matrix)
			print(traceback_matrix)
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
			alignment.add_data(seq_a[curr_element_a - 4:curr_element_a-1], seq_b[curr_element_b - 4:curr_element_b-1])
			curr_element_a -= 4
			curr_element_b -= 4
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


def get_first_key_from_a_dict(a_dict: Dict):
	return next(iter(a_dict))


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray):
	return np.linalg.norm(vector1-vector2)


def points_for_word_embeddings(word_embeddings: Dict[str, np.ndarray], seq1: str, seq2: str):
	return 60/(1 + euclidean_distance(word_embeddings[seq1],  word_embeddings[seq2])) - 15


if __name__ == '__main__':
	from word_embeddings_alignment.data_transformation import read_prot_vec
	# print(create_distance_and_traceback_matrices('ACG', 'ACG', {'ACG':np.array([0, 1])}, 5, 5))
	# print(points_for_word_embeddings(read_prot_vec.read(), 'CGA', 'ACG'))
	create_distance_and_traceback_matrices(
		'ACGA',
		'ACGT',
		read_prot_vec.read(),
		5, 5
	)
