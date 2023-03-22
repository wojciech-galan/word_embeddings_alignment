from word_embeddings_alignment.src.fasta_alignment import FAstaAlignment
from word_embeddings_alignment.src.simple_alignment_representation import SimpleAlignmentRepresentation


def test_align_full_seqs_with_gaps_at_both_ends():
	simple = SimpleAlignmentRepresentation(77, 9, 8)
	simple.add_data('FGHIJKL', 'FGHIJKL')
	simple.set_start_position(3, 2)
	fasta = FAstaAlignment(simple, 'ABCFGHIJKLM', 'DEFGHIJKLNOP', '1', '2')
	assert fasta.aligned_seq1 == '..ABCFGHIJKL...M'
	assert fasta.aligned_seq2 == 'DE...FGHIJKLNOP.'