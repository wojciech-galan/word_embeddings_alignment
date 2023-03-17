from word_embeddings_alignment.src.word_embeddings_water.word_embeddings_water import get_first_key_from_a_dict


def test():
	assert get_first_key_from_a_dict({2: '-', 1: '+'}) == 2
