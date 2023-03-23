from word_embeddings_alignment.src.simple_alignment_representation import SimpleAlignmentRepresentation

GAP_CHAR = '.'


class FastaAlignment(object):

	def __init__(self, simple_alignment: SimpleAlignmentRepresentation, seq1: str, seq2: str, seq1_id: str,
	             seq2_id: str):
		assert simple_alignment.seq1_start_position is not None
		assert simple_alignment.seq2_start_position is not None
		super().__init__()
		self.score = simple_alignment.score
		self.seq1_id = seq1_id
		self.seq2_id = seq2_id
		self.aligned_seq1, self.aligned_seq2 = self.__align_full_seqs_with_gaps_at_both_ends(simple_alignment, seq1, seq2)

	def __align_full_seqs_with_gaps_at_both_ends(self, simple_alignment: SimpleAlignmentRepresentation, seq1: str,
	                                             seq2: str):
		aligned_seq_1 = [simple_alignment.seq2_start_position * GAP_CHAR, seq1[:simple_alignment.seq1_start_position]]
		aligned_seq_2 = [seq2[:simple_alignment.seq2_start_position], simple_alignment.seq1_start_position * GAP_CHAR]
		aligned_seq_1.append(simple_alignment.seq1.replace('-', GAP_CHAR))
		aligned_seq_2.append(simple_alignment.seq2.replace('-', GAP_CHAR))
		chars_of_seq_1_already_used = simple_alignment.seq1_start_position + len(
			simple_alignment.seq1) - simple_alignment.seq1.count('-')
		chars_of_seq_2_already_used = simple_alignment.seq2_start_position + len(
			simple_alignment.seq2) - simple_alignment.seq2.count('-')
		seq1_end_gap_len = len(seq2) - chars_of_seq_2_already_used
		seq2_end_gap_len = len(seq1) - chars_of_seq_1_already_used
		aligned_seq_1.extend([seq1_end_gap_len * GAP_CHAR, seq1[chars_of_seq_1_already_used:]])
		aligned_seq_2.extend([seq2[chars_of_seq_2_already_used:], seq2_end_gap_len * GAP_CHAR])
		return ''.join(aligned_seq_1), ''.join(aligned_seq_2)

	def __str__(self):
		return f'>{self.seq1_id}\r\n{self.aligned_seq1}\r\n>{self.seq2_id}\r\n{self.aligned_seq2}\r\n'
