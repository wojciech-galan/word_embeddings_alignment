import pytest
import blosum as bl

from typing import Dict

from word_embeddings_alignment.regular_water.my_water import align
from word_embeddings_alignment.my_warnings import MultipleEquallyScoredPathsFromMaxTo0
from word_embeddings_alignment.my_warnings import MultipleMaxValuesInDistanceMatrix


def test_not_similar(ednafull_simplified: Dict[str, Dict[str, int]]):
	a = align(
		'AC', 'GT', ednafull_simplified, 5, 5
	)
	assert a.score == 0
	assert a.seq1 == ''
	assert a.seq2 == ''


def test_the_same(ednafull_simplified: Dict[str, Dict[str, int]]):
	a = align(
		'ACG', 'ACG', ednafull_simplified, 5, 5
	)
	assert a.score == 15
	assert a.seq1 == 'ACG'
	assert a.seq2 == 'ACG'


def test_affine_gap_penalty(ednafull_simplified: Dict[str, Dict[str, int]]):
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0):
		a = align(
			'CGCAT', 'CGCCGTAT', ednafull_simplified, 5, 1
		)
	assert a.score == 18
	assert a.seq1 == 'CGC---AT'
	assert a.seq2 == 'CGCCGTAT'


def test_2_affine_gap_penalty(ednafull_simplified: Dict[str, Dict[str, int]]):
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0):
		a = align(
			'ATGGCCTC', 'ACGGCTC', ednafull_simplified, 5, 1
		)
	assert a.score == 21
	assert a.seq1 == 'ATGGCCTC'
	assert a.seq2 == 'ACGG-CTC'


def test_3_affine_gap_penalty(ednafull_simplified: Dict[str, Dict[str, int]]):
	with pytest.warns(MultipleMaxValuesInDistanceMatrix):
		a = align(
			'ATGGCCTC', 'ACGGCTC', ednafull_simplified, 10, 1
		)
	assert a.score == 16
	assert a.seq1 == 'ATGGC'
	assert a.seq2 == 'ACGGC'


def test_4_affine_gap_penalty():
	with pytest.warns(MultipleMaxValuesInDistanceMatrix):
		a = align(
			'CTCTAGCATTAG', 'GTGCACCCA', bl.BLOSUM(62), 10, 1
		)
	assert a.score == 19
	assert a.seq1 == 'GCA'
	assert a.seq2 == 'GCA'
