import pytest

from typing import Dict

from word_embeddings_alignment.regular_water.my_water import align


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
