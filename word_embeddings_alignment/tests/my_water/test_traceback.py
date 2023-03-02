import pytest

import numpy as np

from word_embeddings_alignment.regular_water.my_water import traceback
from word_embeddings_alignment.my_warnings import MultipleEquallyScoredPathsFromMaxTo0
from word_embeddings_alignment.SimpleAlignmentRepresentation import SimpleAlignmentRepresentation


def test_warning():
	with pytest.warns(MultipleEquallyScoredPathsFromMaxTo0, match="Multiple best-scoring alignments are possible"):
		traceback(
			np.array([
				[0, 0, 0],
				[0, 1, 2],
				[0, 2, 3]
			]),
			(2, 2),
			np.array([
				[4, 18],
				[9, 27]
			], dtype=np.byte),
			'__', '__'
		)


def test_mathing_sequences():
	t = traceback(
		np.array([
			[0, 0, 0, 0],
			[0, 5, 0, 0],
			[0, 0, 10, 5],
			[0, 0, 5, 15]
		]),
		(3, 3),
		np.array([
			[4, 18, 0],
			[9, 4, 18],
			[0, 9, 4]
		], dtype=np.byte),
		'ACG', 'ACG'
	)
	assert t.score == 15
	assert t.seq1 == 'ACG'
	assert t.seq2 == 'ACG'