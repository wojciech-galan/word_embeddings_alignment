import pytest
import blosum as bl
import numpy as np

from typing import Dict

from word_embeddings_alignment.src.utils import align
from word_embeddings_alignment.src.my_warnings import MultipleEquallyScoredPathsFromMaxTo0
from word_embeddings_alignment.src.my_warnings import MultipleMaxValuesInDistanceMatrix


def set_up(mocker):
	mocked_points_for_word_embeddings = mocker.patch(
		'word_embeddings_alignment.src.word_embeddings_water.word_embeddings_water.points_for_word_embeddings')
	mocked_points_for_word_embeddings.return_value = 45
	mocked_get_first_key_from_a_dict = mocker.patch(
		'word_embeddings_alignment.src.word_embeddings_water.word_embeddings_water.get_first_key_from_a_dict')
	mocked_get_first_key_from_a_dict.return_value = '___'


def test_not_similar(ednafull_simplified: Dict[str, int]):
	a = next(align(
		'AC', 'GT', ednafull_simplified, 5, 5, 'regular_water'
	))
	assert a.score == 0
	assert a.seq1 == ''
	assert a.seq2 == ''


def test_the_same(ednafull_simplified: Dict[str, int]):
	a = next(align(
		'ACG', 'ACG', ednafull_simplified, 5, 5, 'regular_water'
	))
	assert a.score == 15
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'


def test_affine_gap_penalty(ednafull_simplified: Dict[str, int]):
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0):
		a = next(align(
			'CGCAT', 'CGCCGTAT', ednafull_simplified, 5, 1, 'regular_water'
		))
	assert a.score == 18
	assert a.seq1 == 'CGC---AT'
	assert a.seq2 == 'CGCCGTAT'


def test_2_affine_gap_penalty(ednafull_simplified: Dict[str, int]):
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0):
		a = next(align(
			'ATGGCCTC', 'ACGGCTC', ednafull_simplified, 5, 1, 'regular_water'
		))
	assert a.score == 21
	assert a.seq1 == 'ATGGCCTC'
	assert a.seq2 == 'ACGG-CTC'


def test_3_affine_gap_penalty(ednafull_simplified: Dict[str, int]):
	with pytest.warns(MultipleMaxValuesInDistanceMatrix):
		a = next(align(
			'ATGGCCTC', 'ACGGCTC', ednafull_simplified, 10, 1, 'regular_water'
		))
	assert a.score == 16
	assert a.seq1 == 'ATGGC'
	assert a.seq2 == 'ACGGC'


def test_4_affine_gap_penalty():
	with pytest.warns(MultipleMaxValuesInDistanceMatrix):
		a = next(align(
			'CTCTAGCATTAG', 'GTGCACCCA', bl.BLOSUM(62), 10, 1, 'regular_water'
		))
	assert a.score == 19
	assert a.seq1 == 'GCA'
	assert a.seq2 == 'GCA'


def test_5_affine_gap_penalty():
	a = next(align(
		'AQCHWWL', 'AALLQYL', bl.BLOSUM(62), 10, 1, 'regular_water'
	))
	assert a.score == 6
	assert a.seq1 == 'WL'
	assert a.seq2 == 'YL'


def test_6_affine_gap_penalty():
	a = next(align(
		'DDLDVVAK', 'DDLDTLLGDVVAK', bl.BLOSUM(62), 10, 1, 'regular_water'
	))
	assert a.score == 25
	assert a.seq1 == 'DDLD-----VVAK'
	assert a.seq2 == 'DDLDTLLGDVVAK'


def test_affine_gap_penalty_gaps_in_second_sequence(ednafull_simplified: Dict[str, int]):
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0, match="Multiple best-scoring alignments are possible"):
		a = next(align(
			'CGCCGTAT', 'CGCAT', ednafull_simplified, 5, 1, 'regular_water'
		))
	assert a.score == 18
	assert a.seq1 == 'CGCCGTAT'
	assert a.seq2 == 'CGC---AT'


def test_short_aligned_no_identical_nucleotides(embeddings: Dict[str, np.ndarray]):
	a = next(align(
		'ACG', 'ACG', embeddings, 5, 5, 'word_embeddings_water'
	))
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'
	assert a.score == 45


def test_short_aligned_4_identical_nucleotides(embeddings: Dict[str, np.ndarray], mocker):
	set_up(mocker)
	with pytest.warns(MultipleMaxValuesInDistanceMatrix, match="Multiple best-scoring alignments are possible"):
		a = next(align(
			'AAAA', 'AAAA', embeddings, 5, 5, 'word_embeddings_water'
		))
	assert a.seq1 == 'AAA'
	assert a.seq2 == 'AAA'
	assert a.score == 45


def test_short_aligned_no_identical_nucleotides_return_multiple(embeddings: Dict[str, np.ndarray], mocker):
	set_up(mocker)
	generator = align(
		'AAAA', 'AAAA', embeddings, 5, 5, 'word_embeddings_water',return_multiple_alignments=True
	)
	for a in generator:
		assert a.seq1 == 'AAA'
		assert a.seq2 == 'AAA'
		assert a.score == 45


def test_short_aligned_4_matching_nucleotides(embeddings: Dict[str, np.ndarray], mocker):
	set_up(mocker)
	with pytest.warns(MultipleMaxValuesInDistanceMatrix, match="Multiple best-scoring alignments are possible"):
		a = next(align(
			'ACGA', 'ACGA', embeddings, 5, 5, 'word_embeddings_water'
		))
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'
	assert a.score == 45


def test_short_aligned_4_partially_matching_nucleotides_no_mocks(embeddings: Dict[str, np.ndarray]):
	a = next(align(
		'ACGA', 'ACGT', embeddings, 5, 5, 'word_embeddings_water'
	))
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'
	assert a.score == 45


def test_two_triples_matching_nucleotides_no_mocks(embeddings: Dict[str, np.ndarray]):
	a = next(align(
		'ACGACG', 'ACGACG', embeddings, 5, 5, 'word_embeddings_water'
	))
	assert a.seq1 == 'ACGACG'
	assert a.seq2 == 'ACGACG'
	assert a.score == 90


def test_two_triples_matching_nucleotides_no_mocks_affine_gap_penalty(embeddings: Dict[str, np.ndarray]):
	a = next(align(
		'ACGACG', 'ACGACG', embeddings, 10, 5, 'word_embeddings_water'
	))
	assert a.seq1 == 'ACGACG'
	assert a.seq2 == 'ACGACG'
	assert a.score == 90


def test_two_triples_matching_nucleotides_with_one_gap_no_mocks(embeddings: Dict[str, np.ndarray]):
	a = next(align(
		'ACGACG', 'ACGTACG', embeddings, 5, 5, 'word_embeddings_water'
	))
	assert a.seq1 == 'ACG-ACG'
	assert a.seq2 == 'ACGTACG'
	assert a.score == 85
