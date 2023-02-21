import warnings
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import SupportsFloat, Union

Numeric = Union[SupportsFloat, complex]
UPPER = 1
LEFT = 2
SLANT = 4
AMBIGUOUS_DIRECTIONS = {UPPER | LEFT, UPPER | SLANT, LEFT | SLANT}
GAP = '-'

# EDNAFULL
#     A   T   G   C   S   W   R   Y   K   M   B   V   H   D   N
# A   5  -4  -4  -4  -4   1   1  -4  -4   1  -4  -1  -1  -1  -2
# T  -4   5  -4  -4  -4   1  -4   1   1  -4  -1  -4  -1  -1  -2
# G  -4  -4   5  -4   1  -4   1  -4   1  -4  -1  -1  -4  -1  -2
# C  -4  -4  -4   5   1  -4  -4   1  -4   1  -1  -1  -1  -4  -2
# S  -4  -4   1   1  -1  -4  -2  -2  -2  -2  -1  -1  -3  -3  -1
# W   1   1  -4  -4  -4  -1  -2  -2  -2  -2  -3  -3  -1  -1  -1
# R   1  -4   1  -4  -2  -2  -1  -4  -2  -2  -3  -1  -3  -1  -1
# Y  -4   1  -4   1  -2  -2  -4  -1  -2  -2  -1  -3  -1  -3  -1
# K  -4   1   1  -4  -2  -2  -2  -2  -1  -4  -1  -3  -3  -1  -1
# M   1  -4  -4   1  -2  -2  -2  -2  -4  -1  -3  -1  -1  -3  -1
# B  -4  -1  -1  -1  -1  -3  -3  -1  -1  -3  -1  -2  -2  -2  -1
# V  -1  -4  -1  -1  -1  -3  -1  -3  -3  -1  -2  -1  -2  -2  -1
# H  -1  -1  -4  -1  -3  -1  -3  -1  -3  -1  -2  -2  -1  -2  -1
# D  -1  -1  -1  -4  -3  -1  -1  -3  -1  -3  -2  -2  -2  -1  -1
# N  -2  -2  -2  -2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1

# EDNAFULL_matrix = [
#     [ 5, -4, -4, -4, -4,  1,   1,  -4,  -4,   1,  -4,  -1,  -1,  -1,  -2],
#   	[-4,  5, -4, -4, -4,  1,  -4,   1,   1,  -4,  -1,  -4,  -1,  -1,  -2],
#   	[-4, -4,  5, -4,  1, -4,   1,  -4,   1,  -4,  -1,  -1,  -4,  -1,  -2],
#   	[-4, -4, -4,  5,  1, -4,  -4,   1,  -4,   1,  -1,  -1,  -1,  -4,  -2],
#   	[-4, -4,  1,  1, -1, -4,  -2,  -2,  -2,  -2,  -1,  -1,  -3,  -3,  -1],
#    	[ 1,  1, -4, -4, -4, -1,  -2,  -2,  -2,  -2,  -3,  -3,  -1,  -1,  -1],
#    	[ 1, -4,  1, -4, -2, -2,  -1,  -4,  -2,  -2,  -3,  -1,  -3,  -1,  -1],
#   	[-4,  1, -4,  1, -2, -2,  -4,  -1,  -2,  -2,  -1,  -3,  -1,  -3,  -1],
#   	[-4,  1,  1, -4, -2, -2,  -2,  -2,  -1,  -4,  -1,  -3,  -3,  -1,  -1],
#    	[ 1, -4, -4,  1, -2, -2,  -2,  -2,  -4,  -1,  -3,  -1,  -1,  -3,  -1],
#   	[-4, -1, -1, -1, -1, -3,  -3,  -1,  -1,  -3,  -1,  -2,  -2,  -2,  -1],
#   	[-1, -4, -1, -1, -1, -3,  -1,  -3,  -3,  -1,  -2,  -1,  -2,  -2,  -1],
#   	[-1, -1, -4, -1, -3, -1,  -3,  -1,  -3,  -1,  -2,  -2,  -1,  -2,  -1],
#   	[-1, -1, -1, -4, -3, -1,  -1,  -3,  -1,  -3,  -2,  -2,  -2,  -1,  -1],
#   	[-2, -2, -2, -2, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]
# ]

EDNAFULL_matrix = {'A': {'A': 5,
                         'T': -4,
                         'G': -4,
                         'C': -4,
                         'S': -4,
                         'W': 1,
                         'R': 1,
                         'Y': -4,
                         'K': -4,
                         'M': 1,
                         'B': -4,
                         'V': -1,
                         'H': -1,
                         'D': -1,
                         'N': -2},
                   'T': {'A': -4,
                         'T': 5,
                         'G': -4,
                         'C': -4,
                         'S': -4,
                         'W': 1,
                         'R': -4,
                         'Y': 1,
                         'K': 1,
                         'M': -4,
                         'B': -1,
                         'V': -4,
                         'H': -1,
                         'D': -1,
                         'N': -2},
                   'G': {'A': -4,
                         'T': -4,
                         'G': 5,
                         'C': -4,
                         'S': 1,
                         'W': -4,
                         'R': 1,
                         'Y': -4,
                         'K': 1,
                         'M': -4,
                         'B': -1,
                         'V': -1,
                         'H': -4,
                         'D': -1,
                         'N': -2},
                   'C': {'A': -4,
                         'T': -4,
                         'G': -4,
                         'C': 5,
                         'S': 1,
                         'W': -4,
                         'R': -4,
                         'Y': 1,
                         'K': -4,
                         'M': 1,
                         'B': -1,
                         'V': -1,
                         'H': -1,
                         'D': -4,
                         'N': -2},
                   'S': {'A': -4,
                         'T': -4,
                         'G': 1,
                         'C': 1,
                         'S': -1,
                         'W': -4,
                         'R': -2,
                         'Y': -2,
                         'K': -2,
                         'M': -2,
                         'B': -1,
                         'V': -1,
                         'H': -3,
                         'D': -3,
                         'N': -1},
                   'W': {'A': 1,
                         'T': 1,
                         'G': -4,
                         'C': -4,
                         'S': -4,
                         'W': -1,
                         'R': -2,
                         'Y': -2,
                         'K': -2,
                         'M': -2,
                         'B': -3,
                         'V': -3,
                         'H': -1,
                         'D': -1,
                         'N': -1},
                   'R': {'A': 1,
                         'T': -4,
                         'G': 1,
                         'C': -4,
                         'S': -2,
                         'W': -2,
                         'R': -1,
                         'Y': -4,
                         'K': -2,
                         'M': -2,
                         'B': -3,
                         'V': -1,
                         'H': -3,
                         'D': -1,
                         'N': -1},
                   'Y': {'A': -4,
                         'T': 1,
                         'G': -4,
                         'C': 1,
                         'S': -2,
                         'W': -2,
                         'R': -4,
                         'Y': -1,
                         'K': -2,
                         'M': -2,
                         'B': -1,
                         'V': -3,
                         'H': -1,
                         'D': -3,
                         'N': -1},
                   'K': {'A': -4,
                         'T': 1,
                         'G': 1,
                         'C': -4,
                         'S': -2,
                         'W': -2,
                         'R': -2,
                         'Y': -2,
                         'K': -1,
                         'M': -4,
                         'B': -1,
                         'V': -3,
                         'H': -3,
                         'D': -1,
                         'N': -1},
                   'M': {'A': 1,
                         'T': -4,
                         'G': -4,
                         'C': 1,
                         'S': -2,
                         'W': -2,
                         'R': -2,
                         'Y': -2,
                         'K': -4,
                         'M': -1,
                         'B': -3,
                         'V': -1,
                         'H': -1,
                         'D': -3,
                         'N': -1},
                   'B': {'A': -4,
                         'T': -1,
                         'G': -1,
                         'C': -1,
                         'S': -1,
                         'W': -3,
                         'R': -3,
                         'Y': -1,
                         'K': -1,
                         'M': -3,
                         'B': -1,
                         'V': -2,
                         'H': -2,
                         'D': -2,
                         'N': -1},
                   'V': {'A': -1,
                         'T': -4,
                         'G': -1,
                         'C': -1,
                         'S': -1,
                         'W': -3,
                         'R': -1,
                         'Y': -3,
                         'K': -3,
                         'M': -1,
                         'B': -2,
                         'V': -1,
                         'H': -2,
                         'D': -2,
                         'N': -1},
                   'H': {'A': -1,
                         'T': -1,
                         'G': -4,
                         'C': -1,
                         'S': -3,
                         'W': -1,
                         'R': -3,
                         'Y': -1,
                         'K': -3,
                         'M': -1,
                         'B': -2,
                         'V': -2,
                         'H': -1,
                         'D': -2,
                         'N': -1},
                   'D': {'A': -1,
                         'T': -1,
                         'G': -1,
                         'C': -4,
                         'S': -3,
                         'W': -1,
                         'R': -1,
                         'Y': -3,
                         'K': -1,
                         'M': -3,
                         'B': -2,
                         'V': -2,
                         'H': -2,
                         'D': -1,
                         'N': -1},
                   'N': {'A': -2,
                         'T': -2,
                         'G': -2,
                         'C': -2,
                         'S': -1,
                         'W': -1,
                         'R': -1,
                         'Y': -1,
                         'K': -1,
                         'M': -1,
                         'B': -1,
                         'V': -1,
                         'H': -1,
                         'D': -1,
                         'N': -1}}


class MultipleMaxValuesInDistanceMatrix(UserWarning):
	pass


class MultipleEquallyScoredPathsFromMaxTo0(UserWarning):
	pass


class EmptyAlignment(object):

	def __init__(self):
		super().__init__()
		self.seq1 = []
		self.seq2 = []
		self.seq1_completed = ''
		self.seq2_completed = ''
		self.completed = False

	def set_completed(self):
		self.completed = True
		self.seq1_completed = ''.join(reversed(self.seq1))
		self.seq2_completed = ''.join(reversed(self.seq2))

	def add_data(self, char_a: str, char_b: str):
		self.seq1.append(char_a)
		self.seq2.append(char_b)

	def __str__(self):
		if not self.completed:
			raise RuntimeError("Alignment not completed")
		return '\n'.join([self.seq1_completed, self.seq2_completed])


def align(seq_a: str, seq_b: str, matrix: Dict[str, Dict[str, Numeric]], gap_penalty: Numeric) -> EmptyAlignment:
	distance_matrix, traceback_matrix = create_distance_matrix(seq_a, seq_b, matrix, gap_penalty)
	max_indices_list = find_indices_of_max(distance_matrix)
	if distance_matrix[max_indices_list[0]] == 0:
		alignment = EmptyAlignment()
		alignment.set_completed()
		return alignment
	elif len(max_indices_list) > 1:
		warnings.warn("Multiple best-scoring alignments are possible", MultipleMaxValuesInDistanceMatrix)
	return traceback(distance_matrix, max_indices_list[0], traceback_matrix, seq_a, seq_b)


def create_distance_matrix(seq_a: str, seq_b: str, matrix: Dict[str, Dict[str, Numeric]], gap_penalty: Numeric) -> \
		Tuple[np.ndarray, np.ndarray]:
	# create initial matrix
	distance_matrix = np.full((len(seq_a) + 1, len(seq_b) + 1), np.NaN)
	traceback_matrix = np.zeros((len(seq_a), len(seq_b)), dtype=np.byte)
	# initialize first row and column
	distance_matrix[0, :] = 0
	distance_matrix[:, 0] = 0
	# fill the matrix
	for i, char_a in enumerate(seq_a, 1):
		for j, char_b in enumerate(seq_b, 1):
			slant = distance_matrix[i - 1, j - 1] + matrix[char_a][char_b]
			upper = distance_matrix[i - 1, j] - gap_penalty
			left = distance_matrix[i, j - 1] - gap_penalty
			maximum = max(slant, upper, left, 0)
			distance_matrix[i, j] = maximum
			traceback_matrix[i - 1, j - 1] = (UPPER if maximum == upper else 0) | \
			                                 (LEFT if maximum == left else 0) | \
			                                 (SLANT if maximum == slant else 0)
	return distance_matrix, traceback_matrix


def traceback(distance_matrix: np.ndarray, max_element_indices: Tuple[int], traceback_matrix: np.ndarray,
              seq_a: str, seq_b: str):
	element = distance_matrix[max_element_indices]
	curr_element_a, curr_element_b = max_element_indices
	alignment = EmptyAlignment()
	while element:
		direction = traceback_matrix[curr_element_a - 1, curr_element_b - 1]
		if direction in AMBIGUOUS_DIRECTIONS:
			warnings.warn("Multiple best-scoring alignments are possible", MultipleEquallyScoredPathsFromMaxTo0)
		if direction & SLANT:
			alignment.add_data(seq_a[curr_element_a - 1], seq_b[curr_element_b - 1])
			curr_element_a -= 1
			curr_element_b -= 1
		elif direction & LEFT:
			# gap on the first sequence
			alignment.add_data(GAP, seq_b[curr_element_b - 1])
			curr_element_b -= 1
		else:
			# gap on the second sequence
			alignment.add_data(seq_a[curr_element_a - 1], GAP)
			curr_element_a -= 1
		element = distance_matrix[curr_element_a, curr_element_b]
	alignment.set_completed()
	return alignment


# def traceback_for_forks(distance_matrix: np.ndarray, max_element_index: Tuple[int], traceback_matrix: np.ndarray,
#                         already): pass


def find_indices_of_max(array: np.ndarray) -> List[Tuple[int, int]]:
	maximum = array.max()
	indices = []
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i, j] == maximum:
				indices.append((i, j))
	return indices


print(align('AAATAAC', 'AATATAC', EDNAFULL_matrix, 1))
# print(align('AAATAAA', 'AATATAA', EDNAFULL_matrix, 1))
# print(align('CTCTAGCATTAG', 'GTGCACCCA', EDNAFULL_matrix, 5))

# możemy zapisać info o tym, czy wcześniej była przerwa za pomocą bitu (bitów?) w traceback matrix
