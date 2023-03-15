#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from src import constants
import pickle
from typing import Dict


def read(f_name=constants.PROT_VEC_CSV) -> Dict[str, np.ndarray]:
    ret_dir = {}
    with open(f_name) as f:
        for line in f:
            k, v = line.rstrip().strip('"').split(None, 1)
            v = np.array([float(x) for x in v.split()])
            ret_dir[k] = v
    return ret_dir


if __name__ == '__main__':
    prot_vecs = read()
    with open(constants.PROT_VEC_PICKLE, 'wb') as outfile:
        pickle.dump(prot_vecs, outfile)