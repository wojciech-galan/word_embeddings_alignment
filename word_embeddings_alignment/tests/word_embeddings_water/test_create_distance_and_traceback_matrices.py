import pytest
import numpy as np
from typing import Dict
from word_embeddings_alignment.word_embeddings_water.word_embeddings_water import create_distance_and_traceback_matrices


@pytest.fixture()
def embeddings() -> Dict[str, np.ndarray]:
	return {
		'ACG': np.array([0, 0]),
		'GTA': np.array([0, 1]),
		'TAC': np.array([0, 2]),
		'CGA': np.array([0, 3]),
		'CGT': np.array([0, 4]),
		'GAC': np.array([0, 5])
	}


def set_up(mocker):
	mocked_points_for_word_embeddings = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.points_for_word_embeddings')
	mocked_points_for_word_embeddings.return_value = 45
	mocked_get_first_key_from_a_dict = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.get_first_key_from_a_dict')
	mocked_get_first_key_from_a_dict.return_value = '___'


def test_word_len_not_equal_3(mocker):
	mocked_get_first_key_from_a_dict = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.get_first_key_from_a_dict')
	mocked_get_first_key_from_a_dict.return_value = '____'
	with pytest.raises(NotImplementedError):
		create_distance_and_traceback_matrices(None, None, None, None, None)


def test_short_aligned_no_identical_nucleotides(embeddings: Dict[str, np.ndarray], mocker):
	set_up(mocker)
	dm, tm = create_distance_and_traceback_matrices(
		'ACG',
		'ACG',
		None,
		5, 5
	)
	dm_template = np.array([
		[0, 0, 0,  0],
		[0, 0, 0,  0],
		[0, 0, 0,  0],
		[0, 0, 0, 45]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_short_aligned_4_identical_nucleotides(mocker):
	set_up(mocker)
	dm, tm = create_distance_and_traceback_matrices(
		'AAAA',
		'AAAA',
		None,
		5, 5
	)
	dm_template = np.array([
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0, 45, 45],
		[0, 0, 0, 45, 45]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 4, 4],
		[0, 0, 4, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_short_aligned_4_matching_nucleotides(mocker):
	set_up(mocker)
	dm, tm = create_distance_and_traceback_matrices(
		'ACGA',
		'ACGA',
		None,
		5, 5
	)
	dm_template = np.array([
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0, 45, 45],
		[0, 0, 0, 45, 45]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 4, 4],
		[0, 0, 4, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_short_aligned_4_partially_matching_nucleotides_no_mocks(embeddings: Dict[str, np.ndarray]):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGA',
		'ACGT',
		embeddings,
		5, 5
	)
	dm_template = np.array([
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0,  0,  0],
		[0, 0, 0, 45, 40],
		[0, 0, 0, 40, 35]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 4, 2],
		[0, 0, 1, 3]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_two_triples_matching_nucleotides_no_mocks(embeddings: Dict[str, np.ndarray]):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGACG',
		embeddings,
		5, 5
	)
	dm_template = np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 35, 45],
		[0, 0, 0, 40, 45, 40, 40],
		[0, 0, 0, 35, 40, 45, 40],
		[0, 0, 0, 45, 40, 40, 90]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 1, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_two_triples_matching_nucleotides_higher_score_for_embeddings_similarity(mocker):
	def side_effect(*args):
		if args[1] == args[2]:
			return 45
		return 36

	mocked_points_for_word_embeddings = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.points_for_word_embeddings')
	mocked_points_for_word_embeddings.side_effect = side_effect
	mocked_get_first_key_from_a_dict = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.get_first_key_from_a_dict')
	mocked_get_first_key_from_a_dict.return_value = '___'
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGACG',
		None,
		5, 5
	)

	dm_template = np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 36, 45],
		[0, 0, 0, 40, 45, 40, 40],
		[0, 0, 0, 36, 40, 45, 40],
		[0, 0, 0, 45, 40, 40, 90]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 4, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 4, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_two_triples_matching_nucleotides_slant_or_gap(mocker):
	def side_effect(*args):
		if args[1] == args[2]:
			return 45
		return 35

	mocked_points_for_word_embeddings = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.points_for_word_embeddings')
	mocked_points_for_word_embeddings.side_effect = side_effect
	mocked_get_first_key_from_a_dict = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.get_first_key_from_a_dict')
	mocked_get_first_key_from_a_dict.return_value = '___'
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGACG',
		None,
		5, 5
	)

	dm_template = np.array([
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 35, 45],
		[0, 0, 0, 40, 45, 40, 40],
		[0, 0, 0, 35, 40, 45, 40],
		[0, 0, 0, 45, 40, 40, 90]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 6, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 5, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_two_triples_matching_nucleotides_no_mocks_affine_gap_penalty(embeddings: Dict[str, np.ndarray]):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGACG',
		embeddings,
		10, 5
	)
	dm_template = np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 45, 35, 30, 45],
		[0, 0, 0, 35, 45, 35, 35],
		[0, 0, 0, 30, 35, 45, 35],
		[0, 0, 0, 45, 35, 35, 90]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 4],
		[0, 0, 1, 4, 2, 1],
		[0, 0, 1, 1, 4, 2],
		[0, 0, 4, 2, 1, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)


def test_two_triples_matching_nucleotides_with_one_gap_no_mocks(embeddings: Dict[str, np.ndarray]):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGTACG',
		embeddings,
		5, 5
	)
	dm_template = np.array([
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0,  0,  0,  0,  0,  0],
		[0, 0, 0, 45, 40, 35, 30, 45],
		[0, 0, 0, 40, 35, 30, 25, 40],
		[0, 0, 0, 35, 30, 25, 20, 35],
		[0, 0, 0, 45, 40, 35, 50, 85]
	])
	np.testing.assert_array_equal(dm, dm_template)
	tm_template = np.array([
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0],
		[0, 0, 4, 2, 2, 2, 4],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 1, 3, 3, 3, 1],
		[0, 0, 4, 2, 2, 4, 4]
	])
	np.testing.assert_array_equal(tm, tm_template)
