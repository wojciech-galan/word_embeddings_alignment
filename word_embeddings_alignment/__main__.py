import sys
import argparse
import blosum as bl
from typing import List
from word_embeddings_alignment.src.utils import align
from word_embeddings_alignment.src.utils import read_raw_seq
from word_embeddings_alignment.src.regular_water.matrices.edna_full import EDNAFULL_matrix
from word_embeddings_alignment.data_transformation import read_prot_vec

PROTEIN_EMBEDDINGS = read_prot_vec.read()


def main(args: List[str] = sys.argv[1:]):
	parser = argparse.ArgumentParser(
		description=('''Calculates a local alignment of two sequences using either classic 
		             or word embeddings (default) representation.''')
	)
	parser.add_argument("seq_1", type=str, help='Text file containing raw-formatted sequence 1')
	parser.add_argument("seq_2", type=str, help='Text file containing raw-formatted sequence 2')
	parser.add_argument("gap_open", type=float, help='gap open penalty', default=10)
	parser.add_argument("gap_extend", type=float, help='gap extend penalty', default=1)
	parser.add_argument("dna_or_protein", type=str, help='sequence type', choices=['dna', 'protein'])
	parser.add_argument("--representation", type=str, help='either classic or word embeddings (default) representation',
	                    choices=['classic', 'word_embeddings'], default='word_embeddings')
	parser.add_argument("--multiple", action='store_true', help='''calculate separate alignments for every max value
	 in distance matrix (multiple pairwise alignments are returned)''')
	parsed_args = parser.parse_args(args)
	seq_1 = read_raw_seq(parsed_args.seq_1)
	seq_2 = read_raw_seq(parsed_args.seq_2)
	if parsed_args.dna_or_protein == 'dna' and parsed_args.representation == 'classic':
		for alignment in align(seq_1, seq_2, EDNAFULL_matrix, parsed_args.gap_open, parsed_args.gap_extend,
		                       parsed_args.representation, parsed_args.multiple):
			print(alignment)
	elif parsed_args.dna_or_protein == 'dna' and parsed_args.representation == 'word_embeddings':
		raise NotImplementedError
	elif parsed_args.dna_or_protein == 'protein' and parsed_args.representation == 'classic':
		for alignment in align(seq_1, seq_2, bl.BLOSUM(45), parsed_args.gap_open, parsed_args.gap_extend,
		                       parsed_args.representation, parsed_args.multiple):
			print(alignment)
	else:
		for alignment in align(seq_1, seq_2, PROTEIN_EMBEDDINGS, parsed_args.gap_open, parsed_args.gap_extend,
		                       parsed_args.representation, parsed_args.multiple):
			print(alignment)


if __name__ == "__main__":
	main(sys.argv[1:])
