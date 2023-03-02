import pytest
import numpy as np
from typing import Dict
from word_embeddings_alignment.regular_water.my_water import create_distance_and_traceback_matrices


def test_mathing_sequences__distance_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('ACG', 'ACG', ednafull_simplified, 5, 5)[0],
		np.array([
			[0, 0, 0, 0],
			[0, 5, 0, 0],
			[0, 0, 10, 5],
			[0, 0, 5, 15]
		])
	)


def test_mathing_sequences__traceback_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('ACG', 'ACG', ednafull_simplified, 5, 5)[1],
		np.array([
			[4, 18, 0],
			[9, 4, 18],
			[0, 9, 4]
		], dtype=np.byte)
	)


def test_1_linear_gap__distance_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAAAAA', 'AATATA', ednafull_simplified, 5, 5)[0],
		np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[0, 5, 5, 0, 5, 0, 5],
			[0, 5, 10, 5, 5, 1, 5],
			[0, 5, 10, 6, 10, 5, 6],
			[0, 5, 10, 6, 11, 6, 10],
			[0, 5, 10, 6, 11, 7, 11],
			[0, 5, 10, 6, 11, 7, 12]
		])
	)


def test_1_linear_gap__traceback_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAAAAA', 'AATATA', ednafull_simplified, 5, 5)[1],
		np.array([
			[4, 4, 18, 4, 18, 4],
			[4, 4, 18, 4, 4, 4],
			[4, 4, 4, 4, 18, 4],
			[4, 4, 4, 4, 22, 4],
			[4, 4, 4, 4, 4, 4],
			[4, 4, 4, 4, 4, 4]
		])
	)


def test_2_linear_gap__distance_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAATAAA', 'AATATAA', ednafull_simplified, 1, 1)[0],
		np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 5, 5, 4, 5, 4, 5, 5],
			[0, 5, 10, 9, 9, 8, 9, 10],
			[0, 5, 10, 9, 14, 13, 13, 14],
			[0, 4, 9, 15, 14, 19, 18, 17],
			[0, 5, 9, 14, 20, 19, 24, 23],
			[0, 5, 10, 13, 19, 18, 24, 29],
			[0, 5, 10, 12, 18, 17, 23, 29]
		])
	)


def test_2_linear_gap__traceback_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('AAATAAA', 'AATATAA', ednafull_simplified, 1, 1)[1],
		np.array([
			[4, 4, 18, 4, 18, 4, 4],
			[4, 4, 18, 4, 18, 4, 4],
			[4, 4, 18, 4, 18, 4, 4],
			[9, 9, 4, 18, 4, 18, 18],
			[4, 4, 9, 4, 18, 4, 22],
			[4, 4, 9, 13, 27, 4, 4],
			[4, 4, 9, 13, 27, 13, 4]
		])
	)


def test_3_linear_gap_distance_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 1, 1)[0],
		np.array([
			[0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 5, 4, 5, 5, 4, 3, 2, 1],
			[0, 4, 10, 9, 8, 10, 9, 8, 7],
			[0, 5, 9, 15, 14, 13, 12, 11, 10],
			[0, 4, 8, 14, 13, 12, 11, 17, 16],
			[0, 3, 7, 13, 12, 11, 17, 16, 22]
		])
	)


def test_3_linear_gap__traceback_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 1, 1)[1],
		np.array([
			[4, 18, 4, 4, 18, 18, 18, 18],
			[9, 4, 18, 18, 4, 18, 18, 18],
			[4, 9, 4, 22, 18, 18, 18, 18],
			[9, 9, 9, 27, 27, 27, 4, 18],
			[9, 9, 9, 27, 27, 4, 27, 4]
		])
	)


def test_1_affine_gap_distance_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 5, 1)[0],
		np.array([
			[0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 5, 0, 5, 5, 0, 0, 0, 0],
			[0, 0, 10, 5, 4, 10, 5, 4, 3],
			[0, 5, 5, 15, 10, 9, 8, 7, 6],
			[0, 0, 4, 10, 11, 6, 5, 13, 8],
			[0, 0, 3, 9, 6, 7, 11, 8, 18]
		])
	)


def test_1_affine_gap__traceback_matrix(ednafull_simplified: Dict[str, Dict[str, int]]):
	np.testing.assert_array_equal(
		create_distance_and_traceback_matrices('CGCAT', 'CGCCGTAT', ednafull_simplified, 5, 1)[1],
		np.array([
			[4, 18, 4, 4, 18, 0, 0, 0],
			[9, 4, 18, 18, 4, 18, 18, 18],
			[4, 9, 4, 22, 18, 18, 18, 18],
			[9, 9, 9, 4, 22, 22, 4, 18],
			[0, 9, 9, 13, 4, 4, 9, 4]
		])
	)
