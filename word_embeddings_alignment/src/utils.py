from typing import List, Tuple

import numpy as np


def find_indices_of_max(array: np.ndarray) -> List[Tuple[int, int]]:
	maximum = array.max()
	indices = []
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i, j] == maximum:
				indices.append((i, j))
	return indices
