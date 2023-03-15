import numpy as np
from word_embeddings_alignment.src.word_embeddings_water.word_embeddings_water import euclidean_distance


def test_1a():
	a = np.array([1])
	b = np.array([1])
	assert euclidean_distance(a, b) == 0


def test_1b():
	a = np.array([1])
	b = np.array([3])
	assert euclidean_distance(a, b) == 2


def test_2a():
	a = np.array([0, 1])
	b = np.array([0, 1])
	assert euclidean_distance(a, b) == 0


def test_2b():
	a = np.array([0, 1])
	b = np.array([0, 3])
	assert euclidean_distance(a, b) == 2


def test_2c():
	a = np.array([4, 1])
	b = np.array([0, 4])
	assert euclidean_distance(a, b) == 5
