import pytest
import numpy as np
from word_embeddings_alignment.word_embeddings_water.word_embeddings_water import create_distance_and_traceback_matrices


@pytest.fixture()
def embeddings() -> np.ndarray:
	return {
		'ACG': np.array([0, 0]),
		'CGA': np.array([0, 3]),
		'CGT': np.array([0, 4])
	}


def set_up(mocker):
	mocked_points_for_word_embeddings = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.points_for_word_embeddings')
	mocked_points_for_word_embeddings.return_value = 45
	mocked_get_first_key_from_a_dict = mocker.patch(
		'word_embeddings_alignment.word_embeddings_water.word_embeddings_water.get_first_key_from_a_dict')
	mocked_get_first_key_from_a_dict.return_value = '___'


def test_short_aligned_no_identical_nucleotides(embeddings: np.ndarray, mocker):
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


def test_short_aligned_4_identical_nucleotides(embeddings: np.ndarray, mocker):
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


def test_short_aligned_4_matching_nucleotides(embeddings: np.ndarray, mocker):
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


def test_short_aligned_4_partially_matching_nucleotides_no_mocks(embeddings: np.ndarray):
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
