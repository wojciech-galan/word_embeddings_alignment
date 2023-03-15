import numpy as np
from typing import Dict
from word_embeddings_alignment.word_embeddings_water.word_embeddings_water import traceback


def test_short_aligned_no_identical_nucleotides(
		short_aligned_no_identical_nucleotides_distance_matrix: np.ndarray,
		short_aligned_no_identical_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_no_identical_nucleotides_distance_matrix,
		(3, 3),
		short_aligned_no_identical_nucleotides_traceback_matrix,
		'ACG',
		'ACG'
	)
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'
	assert a.score == 45


def test_short_aligned_4_identical_nucleotides_indices_4_4(
		short_aligned_4_identical_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_identical_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_identical_nucleotides_distance_matrix,
		(4, 4),
		short_aligned_4_identical_nucleotides_traceback_matrix,
		'AAAA',
		'AAAA'
	)
	assert a.seq1 == 'AAA'
	assert a.seq2 == 'AAA'
	assert a.score == 45


def test_short_aligned_4_identical_nucleotides_indices_4_3(
		short_aligned_4_identical_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_identical_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_identical_nucleotides_distance_matrix,
		(4, 3),
		short_aligned_4_identical_nucleotides_traceback_matrix,
		'AAAA',
		'AAAA'
	)
	assert a.seq1 == 'AAA'
	assert a.seq2 == 'AAA'
	assert a.score == 45


def test_short_aligned_4_identical_nucleotides_indices_3_4(
		short_aligned_4_identical_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_identical_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_identical_nucleotides_distance_matrix,
		(3, 4),
		short_aligned_4_identical_nucleotides_traceback_matrix,
		'AAAA',
		'AAAA'
	)
	assert a.seq1 == 'AAA'
	assert a.seq2 == 'AAA'
	assert a.score == 45


def test_short_aligned_4_identical_nucleotides_indices_3_3(
		short_aligned_4_identical_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_identical_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_identical_nucleotides_distance_matrix,
		(3, 3),
		short_aligned_4_identical_nucleotides_traceback_matrix,
		'AAAA',
		'AAAA'
	)
	assert a.seq1 == 'AAA'
	assert a.seq2 == 'AAA'
	assert a.score == 45


def test_short_aligned_4_matching_nucleotides_indices_4_4(
		short_aligned_4_matching_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_matching_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_matching_nucleotides_distance_matrix,
		(4, 4),
		short_aligned_4_matching_nucleotides_traceback_matrix,
		'ACGA',
		'ACGA'
	)
	assert a.seq1 == 'CGA'
	assert a.seq2 == 'CGA'
	assert a.score == 45


def test_short_aligned_4_matching_nucleotides_indices_4_3(
		short_aligned_4_matching_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_matching_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_matching_nucleotides_distance_matrix,
		(4, 3),
		short_aligned_4_matching_nucleotides_traceback_matrix,
		'ACGA',
		'ACGA'
	)
	assert a.seq1 == 'CGA'
	assert a.seq2 == 'ACG'
	assert a.score == 45


def test_short_aligned_4_matching_nucleotides_indices_3_4(
		short_aligned_4_matching_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_matching_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_matching_nucleotides_distance_matrix,
		(3, 4),
		short_aligned_4_matching_nucleotides_traceback_matrix,
		'ACGA',
		'ACGA'
	)
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'CGA'
	assert a.score == 45


def test_short_aligned_4_matching_nucleotides_indices_3_3(
		short_aligned_4_matching_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_matching_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_matching_nucleotides_distance_matrix,
		(3, 3),
		short_aligned_4_matching_nucleotides_traceback_matrix,
		'ACGA',
		'ACGA'
	)
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'
	assert a.score == 45


def test_short_aligned_4_partially_matching_nucleotides(
		short_aligned_4_partially_matching_nucleotides_distance_matrix: np.ndarray,
		short_aligned_4_partially_matching_nucleotides_traceback_matrix: np.ndarray
):
	a = traceback(
		short_aligned_4_partially_matching_nucleotides_distance_matrix,
		(3, 3),
		short_aligned_4_partially_matching_nucleotides_traceback_matrix,
		'ACGA',
		'ACGT'
	)
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'
	assert a.score == 45


def test_two_triples_matching_nucleotides(
		two_triples_matching_nucleotides_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_no_mocks_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_no_mocks_distance_matrix,
		(6, 6),
		two_triples_matching_nucleotides_no_mocks_traceback_matrix,
		'ACGACG',
		'ACGACG'
	)
	assert a.seq1 == 'ACGACG'
	assert a.seq2 == 'ACGACG'
	assert a.score == 90


def test_two_triples_matching_nucleotides_higher_score_for_embeddings_similarity(
		two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_traceback_matrix: np.ndarray,
):
	a = traceback(
		two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_distance_matrix,
		(6, 6),
		two_triples_matching_nucleotides_higher_score_for_embeddings_similarity_traceback_matrix,
		'ACGACG',
		'ACGACG'
	)
	assert a.seq1 == 'ACGACG'
	assert a.seq2 == 'ACGACG'
	assert a.score == 90


def test_two_triples_matching_nucleotides_slant_or_gap(
		two_triples_matching_nucleotides_slant_or_gap_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_slant_or_gap_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_slant_or_gap_distance_matrix,
		(6, 6),
		two_triples_matching_nucleotides_slant_or_gap_traceback_matrix,
		'ACGACG',
		'ACGACG'
	)
	assert a.seq1 == 'ACGACG'
	assert a.seq2 == 'ACGACG'
	assert a.score == 90


def test_two_triples_matching_nucleotides_no_mocks_affine_gap_penalty(
		two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_distance_matrix,
		(6, 6),
		two_triples_matching_nucleotides_no_mocks_affine_gap_penalty_traceback_matrix,
		'ACGACG',
		'ACGACG'
	)
	assert a.seq1 == 'ACGACG'
	assert a.seq2 == 'ACGACG'
	assert a.score == 90


def test_two_triples_matching_nucleotides_with_one_gap(
		two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_no_mocks_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix,
		(6, 7),
		two_triples_matching_nucleotides_with_one_gap_no_mocks_traceback_matrix,
		'ACGACG',
		'ACGTACG'
	)
	assert a.seq1 == 'ACG-ACG'
	assert a.seq2 == 'ACGTACG'
	assert a.score == 85


def test_two_triples_matching_nucleotides_with_one_gap_no_mocks_swapped_seqs(
		two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_no_mocks_swapped_seqs_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_one_gap_no_mocks_distance_matrix.T,
		(7, 6),
		two_triples_matching_nucleotides_with_one_gap_no_mocks_swapped_seqs_traceback_matrix,
		'ACGTACG',
		'ACGACG'
	)
	assert a.seq1 == 'ACGTACG'
	assert a.seq2 == 'ACG-ACG'
	assert a.score == 85


def test_two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_distance_matrix,
		(6, 7),
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_end_no_mocks_traceback_matrix,
		'ACGACGAC',
		'ACGTACG'
	)
	assert a.seq1 == 'ACG-ACG'
	assert a.seq2 == 'ACGTACG'
	assert a.score == 85


def test_two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_distance_matrix,
		(8, 7),
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_traceback_matrix,
		'CGACGACG',
		'ACGTACG'
	)
	assert a.seq1 == 'ACG-ACG'
	assert a.seq2 == 'ACGTACG'
	assert a.score == 85


def test_two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_affine_gap_score(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix,
		(8, 7),
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_traceback_matrix,
		'CGACGACG',
		'ACGTACG'
	)
	assert a.seq1 == 'ACG-ACG'
	assert a.seq2 == 'ACGTACG'
	assert a.score == 80


def test_two_triples_matching_nucleotides_with_one_gap_add_chars_at_the_beginning_affine_gap_score_swapped(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_one_gap_add_chars_at_the_beginning_no_mocks_affine_gap_score_swapped_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_one_gap_additional_chars_at_the_beginning_no_mocks_affine_gap_score_distance_matrix.T,
		(7, 8),
		two_triples_matching_nucleotides_with_one_gap_add_chars_at_the_beginning_no_mocks_affine_gap_score_swapped_traceback_matrix,
		'ACGTACG',
		'CGACGACG'
	)
	assert a.seq1 == 'ACGTACG'
	assert a.seq2 == 'ACG-ACG'
	assert a.score == 80


def test_two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap(
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_distance_matrix,
		(8, 6),
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_traceback_matrix,
		'ACGTTACG',
		'ACGACG'
	)
	assert a.seq1 == 'ACGTTACG'
	assert a.seq2 == 'ACG--ACG'
	assert a.score == 75


def test_two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_swapped(
		embeddings: Dict[str, np.ndarray],
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_distance_matrix: np.ndarray,
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_swapped_traceback_matrix: np.ndarray
):
	a = traceback(
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_distance_matrix.T,
		(6, 8),
		two_triples_matching_nucleotides_with_longer_gap_no_mocks_affine_gap_swapped_traceback_matrix,
		'ACGACG',
		'ACGTTACG'
	)
	assert a.seq1 == 'ACG--ACG'
	assert a.seq2 == 'ACGTTACG'
	assert a.score == 75
