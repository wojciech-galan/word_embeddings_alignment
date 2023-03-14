import pytest
from typing import Dict
import numpy as np


@pytest.fixture(scope="module")
def embeddings() -> Dict[str, np.ndarray]:
	return {
		'ACG': np.array([0, 0]),
		'GTA': np.array([0, 1]),
		'TAC': np.array([0, 2]),
		'CGA': np.array([0, 3]),
		'CGT': np.array([0, 4]),
		'GAC': np.array([0, 5]),
		'GTT': np.array([0, 7]),
		'TTA': np.array([0, 9])
	}


@pytest.fixture(scope='module')
def short_aligned_no_identical_nucleotides_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0],
		[0, 0, 0,  0],
		[0, 0, 0,  0],
		[0, 0, 0, 45]
	])


@pytest.fixture(scope='module')
def short_aligned_no_identical_nucleotides_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 4]
	])


@pytest.fixture(scope='module')
def short_aligned_4_matching_nucleotides_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0, 45, 45],
		[0, 0, 0, 45, 45]
	])


@pytest.fixture(scope='module')
def short_aligned_4_matching_nucleotides_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 4, 4],
		[0, 0, 4, 4]
	])


@pytest.fixture(scope='module')
def short_aligned_4_identical_nucleotides_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0, 45, 45],
		[0, 0, 0, 45, 45]
	])


@pytest.fixture(scope='module')
def short_aligned_4_identical_nucleotides_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 4, 4],
		[0, 0, 4, 4]
	])


@pytest.fixture(scope='module')
def short_aligned_4_partially_matching_nucleotides_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0, 45, 40],
		[0, 0, 0, 40, 35]
	])


@pytest.fixture(scope='module')
def short_aligned_4_partially_matching_nucleotides_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 4, 2],
		[0, 0, 1, 3]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_no_mocks_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 35, 45],
		[0, 0, 0, 40, 45, 40, 40],
		[0, 0, 0, 35, 40, 45, 40],
		[0, 0, 0, 45, 40, 40, 90]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_no_mocks_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 1, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 36, 45],
		[0, 0, 0, 40, 45, 40, 40],
		[0, 0, 0, 36, 40, 45, 40],
		[0, 0, 0, 45, 40, 40, 90]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 4, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 4, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_slant_or_gap_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 35, 45],
		[0, 0, 0, 40, 45, 40, 40],
		[0, 0, 0, 35, 40, 45, 40],
		[0, 0, 0, 45, 40, 40, 90]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_slant_or_gap_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 6, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 5, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 35, 30, 45],
		[0, 0, 0, 35, 45, 35, 35],
		[0, 0, 0, 30, 35, 45, 35],
		[0, 0, 0, 45, 35, 35, 90]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 1, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 35, 30, 45],
		[0, 0, 0, 40, 35, 30, 25, 40],
		[0, 0, 0, 35, 30, 25, 20, 35],
		[0, 0, 0, 45, 40, 35, 50, 85]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_no_mocks_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 2, 4],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 4, 2, 2, 4, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_no_mocks_swapped_seqs_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 4],
		[0, 0, 1, 3, 3, 1],
		[0, 0, 1, 3, 3, 1],
		[0, 0, 1, 3, 3, 4],
		[0, 0, 4, 2, 2, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 35, 30, 45],
		[0, 0, 0, 40, 35, 30, 25, 40],
		[0, 0, 0, 35, 30, 25, 20, 35],
		[0, 0, 0, 45, 40, 35, 50, 85],
		[0, 0, 0, 40, 35, 30, 55, 80],
		[0, 0, 0, 35, 30, 25, 50, 75]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 2, 4],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 4, 2, 2, 4, 4],
		[0, 0, 1, 3, 3, 4, 1],
		[0, 0, 1, 3, 3, 1, 1]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0, 15, 10, 15, 10],
		[0, 0, 0,  0, 15, 10, 10,  5],
		[0, 0, 0, 45, 40, 35, 30, 45],
		[0, 0, 0, 40, 35, 30, 25, 40],
		[0, 0, 0, 35, 30, 25, 20, 35],
		[0, 0, 0, 45, 40, 35, 50, 85]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 4, 2, 4, 2],
		[0, 0, 0, 4, 2, 1, 3],
		[0, 0, 4, 2, 2, 2, 4],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 4, 2, 2, 4, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0, 15,  5, 15,  5],
		[0, 0, 0,  0, 15,  5,  5,  0],
		[0, 0, 0, 45, 35, 34, 33, 45],
		[0, 0, 0, 35, 25, 24, 23, 35],
		[0, 0, 0, 34, 24, 23, 22, 34],
		[0, 0, 0, 45, 35, 34, 50, 80]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 4, 6, 4, 2],
		[0, 0, 0, 4, 2, 1, 0],
		[0, 0, 4, 2, 2, 2, 4],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 4, 2, 2, 4, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_one_gap_add_chars_at_the_beginning_no_mocks_affine_gap_score_swapped_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 4, 2, 2, 4],
		[0, 0, 4, 4, 1, 3, 3, 1],
		[0, 0, 5, 1, 1, 3, 3, 1],
		[0, 0, 4, 2, 1, 3, 3, 4],
		[0, 0, 1, 0, 4, 2, 2, 4]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_distance_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 35, 30, 45],
		[0, 0, 0, 35, 25, 20, 35],
		[0, 0, 0, 30, 20, 15, 30],
		[0, 0, 0, 25, 15, 10, 36],
		[0, 0, 0, 20, 15,  5, 40],
		[0, 0, 0, 45, 35, 30, 75]
	])


@pytest.fixture(scope='module')
def two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_traceback_matrix() -> np.ndarray:
	return np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 4],
		[0, 0, 1, 3, 3, 1],
		[0, 0, 1, 3, 3, 1],
		[0, 0, 1, 3, 3, 4],
		[0, 0, 1, 4, 3, 4],
		[0, 0, 4, 2, 2, 4]
	])