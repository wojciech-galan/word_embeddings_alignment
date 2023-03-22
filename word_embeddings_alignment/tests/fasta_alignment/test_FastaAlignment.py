from word_embeddings_alignment.src.fasta_alignment import FAstaAlignment
from word_embeddings_alignment.src.simple_alignment_representation import SimpleAlignmentRepresentation


def test_align_full_seqs_with_gaps_at_both_ends():
	simple = SimpleAlignmentRepresentation(77, 9, 8)
	simple.add_data('FGHIJKL', 'FGHIJKL')
	simple.set_start_position(3, 2)
	fasta = FAstaAlignment(simple, 'ABCFGHIJKLM', 'DEFGHIJKLNOP', '1', '2')
	assert fasta.aligned_seq1 == '..ABCFGHIJKL...M'
	assert fasta.aligned_seq2 == 'DE...FGHIJKLNOP.'


def test_align_full_seqs_with_gaps_at_both_ends_and_inside_first():
	simple = SimpleAlignmentRepresentation(77, 9, 7)
	simple.add_data('FGHIJKL', 'FGH-JKL')
	simple.set_start_position(3, 2)
	fasta = FAstaAlignment(simple, 'ABCFGHIJKLM', 'DEFGHJKLNOP', '1', '2')
	assert fasta.aligned_seq1 == '..ABCFGHIJKL...M'
	assert fasta.aligned_seq2 == 'DE...FGH.JKLNOP.'


def test_align_full_seqs_with_gaps_at_both_ends_and_inside_second():
	simple = SimpleAlignmentRepresentation(77, 7, 8)
	simple.add_data('FGH--KL', 'FGHIJKL')
	simple.set_start_position(3, 2)
	fasta = FAstaAlignment(simple, 'ABCFGHKLM', 'DEFGHIJKLNOP', '1', '2')
	assert fasta.aligned_seq1 == '..ABCFGH..KL...M'
	assert fasta.aligned_seq2 == 'DE...FGHIJKLNOP.'


def test_align_full_seqs_with_gaps_at_both_ends_and_inside_both():
	simple = SimpleAlignmentRepresentation(77, 8, 7)
	simple.add_data('FGHI-KL', 'FGH-JKL')
	simple.set_start_position(3, 2)
	fasta = FAstaAlignment(simple, 'ABCFGHIKLM', 'DEFGHJKLNOP', '1', '2')
	assert fasta.aligned_seq1 == '..ABCFGHI.KL...M'
	assert fasta.aligned_seq2 == 'DE...FGH.JKLNOP.'