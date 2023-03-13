import pytest
import numpy as np
from typing import Dict
from word_embeddings_alignment.regular_water.regular_water import create_distance_and_traceback_matrices


def test_mathing_sequences__distance_matrix(ednafull_simplified: Dict[str, int]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('ACG', 'ACG', ednafull_simplified, 5, 5)[0],
		np.array([
			[0, 0, 0, 0],
			[0, 5, 0, 0],
			[0, 0, 10, 5],
			[0, 0, 5, 15]
		])
	)


def test_mathing_sequences__traceback_matrix(ednafull_simplified: Dict[str, int]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('ACG', 'ACG', ednafull_simplified, 5, 5)[1],
		np.array([
			[4, 2, 0],
			[1, 4, 2],
			[0, 1, 4]
		], dtype=np.byte)
	)


def test_1_linear_gap__distance_matrix(ednafull_simplified: Dict[str, int], linear_gap__distance_matrix_1: np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAAAAA', 'AATATA', ednafull_simplified, 5, 5)[0],
		linear_gap__distance_matrix_1
	)


def test_1_linear_gap__traceback_matrix(ednafull_simplified: Dict[str, int], linear_gap__traceback_matrix_1: np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAAAAA', 'AATATA', ednafull_simplified, 5, 5)[1],
		linear_gap__traceback_matrix_1
	)


def test_2_linear_gap__distance_matrix(ednafull_simplified: Dict[str, int], linear_gap__distance_matrix_2: np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAATAAA', 'AATATAA', ednafull_simplified, 1, 1)[0],
		linear_gap__distance_matrix_2
	)


def test_2_linear_gap__traceback_matrix(ednafull_simplified: Dict[str, int], linear_gap__traceback_matrix_2: np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAATAAA', 'AATATAA', ednafull_simplified, 1, 1)[1],
		linear_gap__traceback_matrix_2
	)


def test_3_linear_gap_distance_matrix(ednafull_simplified: Dict[str, int], linear_gap__distance_matrix_3:np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 1, 1)[0],
		linear_gap__distance_matrix_3
	)


def test_3_linear_gap__traceback_matrix(ednafull_simplified: Dict[str, int], linear_gap__traceback_matrix_3: np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 1, 1)[1],
		linear_gap__traceback_matrix_3
	)


def test_1_affine_gap_distance_matrix(ednafull_simplified: Dict[str, int], affine_gap__distance_matrix_1: np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 5, 1)[0],
		affine_gap__distance_matrix_1
	)


def test_1_affine_gap__traceback_matrix(ednafull_simplified: Dict[str, int], affine_gap__traceback_matrix_1: np.ndarray):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 5, 1)[1],
		affine_gap__traceback_matrix_1
	)
