import copy
import pytest
from word_embeddings_alignment.src.simple_alignment_representation import SimpleAlignmentRepresentation


@pytest.fixture(scope="module")
def simple_alignment() -> SimpleAlignmentRepresentation:
	s = SimpleAlignmentRepresentation(9)
	s._seq1 = ['C', 'T', 'A']
	s._seq2 = ['C', '-', 'A']
	return s


def test_internal_data(simple_alignment: SimpleAlignmentRepresentation):
	assert simple_alignment._score == 9


def test_properties(simple_alignment: SimpleAlignmentRepresentation):
	assert simple_alignment.seq1 == 'ATC'
	assert simple_alignment.seq2 == 'A-C'
	assert simple_alignment.score == 9


def test_add_data(simple_alignment: SimpleAlignmentRepresentation):
	simple_alignment_copy = copy.deepcopy(simple_alignment)
	simple_alignment_copy.add_data('bf', 'gf')
	assert simple_alignment_copy.seq1 == 'bfATC'
	assert simple_alignment_copy.seq2 == 'gfA-C'


def test_eq_different_type(simple_alignment: SimpleAlignmentRepresentation):
	assert (simple_alignment == 9) is False


def test_eq_diff_score(simple_alignment: SimpleAlignmentRepresentation):
	simple_alignment_copy = copy.deepcopy(simple_alignment)
	simple_alignment_copy._score = 77
	assert (simple_alignment == simple_alignment_copy) is False


def test_eq_diff_seq1(simple_alignment: SimpleAlignmentRepresentation):
	simple_alignment_copy = copy.deepcopy(simple_alignment)
	simple_alignment_copy._seq1 = ['C', 'T', 'a']
	assert (simple_alignment == simple_alignment_copy) is False


def test_eq_equal_elements(simple_alignment: SimpleAlignmentRepresentation):
	simple_alignment_copy = copy.deepcopy(simple_alignment)
	# assert they are two alignments, not two references to the same alignment
	assert simple_alignment is not simple_alignment_copy
	# assert they are equal
	assert simple_alignment == simple_alignment_copy


def test_str(simple_alignment: SimpleAlignmentRepresentation):
	assert str(simple_alignment) == 'ATC\nA-C\n9'
