import pytest
import numpy as np
from typing import Dict
from word_embeddings_alignment.word_embeddings_water.word_embeddings_water import create_distance_and_traceback_matrices


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


def test_short_aligned_no_identical_nucleotides(embeddings: Dict[str, np.ndarray],
                                                short_aligned_no_identical_nucleotides_distance_matrix: np.ndarray,
                                                short_aligned_no_identical_nucleotides_traceback_matrix: np.ndarray,
                                                mocker):
	set_up(mocker)
	dm, tm = create_distance_and_traceback_matrices(
		'ACG',
		'ACG',
		None,
		5, 5
	)
	np.testing.assert_array_equal(dm, short_aligned_no_identical_nucleotides_distance_matrix)
	np.testing.assert_array_equal(tm, short_aligned_no_identical_nucleotides_traceback_matrix)


def test_short_aligned_4_identical_nucleotides(short_aligned_4_identical_nucleotides_distance_matrix: np.ndarray,
                                               short_aligned_4_identical_nucleotides_traceback_matrix: np.ndarray,
                                               mocker):
	set_up(mocker)
	dm, tm = create_distance_and_traceback_matrices(
		'AAAA',
		'AAAA',
		None,
		5, 5
	)
	np.testing.assert_array_equal(dm, short_aligned_4_identical_nucleotides_distance_matrix)
	np.testing.assert_array_equal(tm, short_aligned_4_identical_nucleotides_traceback_matrix)


def test_short_aligned_4_matching_nucleotides(short_aligned_4_matching_nucleotides_distance_matrix: np.ndarray,
                                              short_aligned_4_matching_nucleotides_traceback_matrix: np.ndarray,
                                              mocker):
	set_up(mocker)
	dm, tm = create_distance_and_traceback_matrices(
		'ACGA',
		'ACGA',
		None,
		5, 5
	)
	np.testing.assert_array_equal(dm, short_aligned_4_matching_nucleotides_distance_matrix)
	np.testing.assert_array_equal(tm, short_aligned_4_matching_nucleotides_traceback_matrix)


def test_short_aligned_4_partially_matching_nucleotides_no_mocks(embeddings: Dict[str, np.ndarray],
                                                                 short_aligned_4_partially_matching_nucleotides_distance_matrix: np.ndarray,
                                                                 short_aligned_4_partially_matching_nucleotides_traceback_matrix: np.ndarray):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGA',
		'ACGT',
		embeddings,
		5, 5
	)
	np.testing.assert_array_equal(dm, short_aligned_4_partially_matching_nucleotides_distance_matrix)
	np.testing.assert_array_equal(tm, short_aligned_4_partially_matching_nucleotides_traceback_matrix)


def test_two_triples_matching_nucleotides_no_mocks(embeddings: Dict[str, np.ndarray],
                                                   two_triples_matching_nucleotides_no_mocks_distance_matrix: np.ndarray,
                                                   two_triples_matching_nucleotides_no_mocks_traceback_matrix: np.ndarray):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGACG',
		embeddings,
		5, 5
	)
	np.testing.assert_array_equal(dm, two_triples_matching_nucleotides_no_mocks_distance_matrix)
	np.testing.assert_array_equal(tm, two_triples_matching_nucleotides_no_mocks_traceback_matrix)


def test_two_triples_matching_nucleotides_higher_score_for_embeddings_similarity(
		two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_traceback_matrix: np.ndarray,
		mocker
):
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
	np.testing.assert_array_equal(dm,
	                              two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_distance_matrix)
	np.testing.assert_array_equal(tm,
	                              two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_traceback_matrix)


def test_two_triples_matching_nucleotides_slant_or_gap(
		mocker,
		two_triples_matching_nucleotides_slant_or_gap_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_slant_or_gap_traceback_matrix: np.ndarray
):
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
	np.testing.assert_array_equal(dm, two_triples_matching_nucleotides_slant_or_gap_distance_matrix)
	np.testing.assert_array_equal(tm, two_triples_matching_nucleotides_slant_or_gap_traceback_matrix)


def test_two_triples_matching_nucleotides_no_mocks_affine_gap_penalty(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGACG',
		embeddings,
		10, 5
	)
	np.testing.assert_array_equal(dm, two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_distance_matrix)
	np.testing.assert_array_equal(tm, two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_traceback_matrix)


def test_two_triples_matching_nucleotides_with_one_gap_no_mocks(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_no_mocks_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACG',
		'ACGTACG',
		embeddings,
		5, 5
	)
	np.testing.assert_array_equal(dm, two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix)
	np.testing.assert_array_equal(tm, two_triples_matching_nucleotides_with_one_gap_no_mocks_traceback_matrix)


def test_two_triples_matching_nucleotides_with_one_gap_no_mocks_swapped_seqs(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_no_mocks_swapped_seqs_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGTACG',
		'ACGACG',
		embeddings,
		5, 5
	)
	np.testing.assert_array_equal(dm, two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix.T)
	np.testing.assert_array_equal(tm,
	                              two_triples_matching_nucleotides_with_one_gap_no_mocks_swapped_seqs_traceback_matrix)


def test_two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGACGAC',
		'ACGTACG',
		embeddings,
		5, 5
	)
	np.testing.assert_array_equal(dm,
	                              two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_distance_matrix)
	np.testing.assert_array_equal(tm,
	                              two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_traceback_matrix)


def test_two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'CGACGACG',
		'ACGTACG',
		embeddings,
		5, 5
	)
	np.testing.assert_array_equal(dm,
	                              two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_distance_matrix)
	np.testing.assert_array_equal(tm,
	                              two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_traceback_matrix)


def test_two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'CGACGACG',
		'ACGTACG',
		embeddings,
		10, 1
	)
	np.testing.assert_array_equal(dm,
	                              two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix)
	np.testing.assert_array_equal(tm,
	                              two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_traceback_matrix)


def test_two_triples_matching_nucleotides_with_one_gap_add_chars_at_the_beginning_no_mocks_affine_gap_score_swapped(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_add_chars_at_the_beginning_no_mocks_affine_gap_score_swapped_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGTACG',
		'CGACGACG',
		embeddings,
		10, 1
	)
	np.testing.assert_array_equal(dm,
	                              two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix.T)
	np.testing.assert_array_equal(tm,
	                              two_triples_matching_nucleotides_with_one_gap_add_chars_at_the_beginning_no_mocks_affine_gap_score_swapped_traceback_matrix)


def test_two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap(
embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_traceback_matrix: np.ndarray
):
	dm, tm = create_distance_and_traceback_matrices(
		'ACGTTACG',
		'ACGACG',
		embeddings,
		10, 5
	)
	np.testing.assert_array_equal(dm,
	                              two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_distance_matrix)
	np.testing.assert_array_equal(tm,
	                              two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_traceback_matrix)