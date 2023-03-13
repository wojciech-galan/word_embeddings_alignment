import blosum as bl
from word_embeddings_alignment.regular_water.my_water import align
from word_embeddings_alignment.regular_water.matrices.edna_full import EDNAFULL_matrix

if __name__ == "__main__":
	pass
	print(align('AAATAAC', 'AATATAC', EDNAFULL_matrix, 5, 1))
	print(align('AAATAAA', 'AATATAA', EDNAFULL_matrix, 1, 1))
	print(align('ACGTCTGATACGCCGTATAGTCTATCT', 'CTGATTCGCATCGTCTATCT', EDNAFULL_matrix, 5, 1))
	print(align('CGCAT', 'CGCCGTAT', EDNAFULL_matrix, 5, 1))
	print(align('CTCTAGCATTAG', 'GTGCACCCA', bl.BLOSUM(62), 10, 1))
	print(align('DDLDVVAK', 'DDLDTLLGDVVAK', bl.BLOSUM(62), 10, 1))
	print(align('DDLDVVAK', 'DDLTLLGDVVAK', bl.BLOSUM(62), 10, 1))