import pytest

import numpy as np

from word_embeddings_alignment.regular_water.my_water import traceback
from word_embeddings_alignment.my_warnings import MultipleEquallyScoredPathsFromMaxTo0


def test_warning():
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0, match="Multiple best-scoring alignments are possible"):
		traceback(
			np.array([
				[0, 0, 0],
				[0, 1, 2],
				[0, 2, 3]
			]),
			(2, 2),
			np.array([
				[4, 18],
				[9, 27]
			], dtype=np.byte),
			'__', '__'
		)


def test_mathing_sequences():
	a = traceback(
		np.array([
			[0, 0, 0, 0],
			[0, 5, 0, 0],
			[0, 0, 10, 5],
			[0, 0, 5, 15]
		]),
		(3, 3),
		np.array([
			[4, 18, 0],
			[9, 4, 18],
			[0, 9, 4]
		], dtype=np.byte),
		'ACG', 'ACG'
	)
	assert a.score == 15
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'


def test_1_linear_gap(linear_gap__distance_matrix_1: np.ndarray, linear_gap__traceback_matrix_1: np.ndarray):
	a = traceback(
		linear_gap__distance_matrix_1,
		(6, 6),
		linear_gap__traceback_matrix_1,
		'AAAAAA', 'AATATA'
	)
	assert a.score == 12
	assert a.seq1 == 'AAAAAA'
	assert a.seq2 == 'AATATA'


def test_2_linear_gap(linear_gap__distance_matrix_2: np.ndarray, linear_gap__traceback_matrix_2: np.ndarray):
	a = traceback(
		linear_gap__distance_matrix_2,
		(7, 7),
		linear_gap__traceback_matrix_2,
		'AAATAAA', 'AATATAA'
	)
	assert a.score == 29
	assert a.seq1 == 'AATA-AA'
	assert a.seq2 == 'AATATAA'


def test_3_linear_gap(linear_gap__distance_matrix_3: np.ndarray, linear_gap__traceback_matrix_3: np.ndarray):
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0, match="Multiple best-scoring alignments are possible"):
		a = traceback(
			linear_gap__distance_matrix_3,
			(5, 8),
			linear_gap__traceback_matrix_3,
			'CGCAT', 'CGCCGTAT'
		)
	assert a.score == 22
	assert a.seq1 == 'CGC---AT'
	assert a.seq2 == 'CGCCGTAT'


def test_1_affine_gap(affine_gap__distance_matrix_1: np.ndarray, affine_gap__traceback_matrix_1: np.ndarray):
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0, match="Multiple best-scoring alignments are possible"):
		a = traceback(
			affine_gap__distance_matrix_1,
			(5, 8),
			affine_gap__traceback_matrix_1,
			'CGCAT', 'CGCCGTAT'
		)
	assert a.score == 18
	assert a.seq1 == 'CGC---AT'
	assert a.seq2 == 'CGCCGTAT'
