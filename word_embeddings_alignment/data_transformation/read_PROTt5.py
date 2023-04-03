import h5py
from word_embeddings_alignment.src import constants
from word_embeddings_alignment.src.utils import read_fasta_seq
from word_embeddings_alignment.src.word_embeddings_water.word_embeddings_water import cosine_similarity
import pickle
from typing import Dict

'''
Reads data from https://github.com/agemagician/ProtTrans and transforms it into a distance matrix
'''


def read(vec_f_name=constants.PROTT5_VEC, aa_f_name=constants.PROTT5_AA) -> Dict[str, float]:
	similarities = {}
	embeddings = {}
	_, aminoacids = next(read_fasta_seq(aa_f_name))
	with h5py.File(vec_f_name) as f:
		vectors = f['aminoacids']
		for i, aa in enumerate(aminoacids):
			embeddings[aa] = vectors[i]
	for i, aa1 in enumerate(aminoacids):
		for aa2 in aminoacids[i:]:
			similarity = cosine_similarity(embeddings[aa1], embeddings[aa2])
			similarities[aa1 + aa2] = similarity
			similarities[aa2 + aa1] = similarity
	return similarities


if __name__ == '__main__':
	prot_similarities = read()
	with open(constants.PROTT5_VEC_PICKLE, 'wb') as outfile:
		pickle.dump(prot_similarities, outfile)
