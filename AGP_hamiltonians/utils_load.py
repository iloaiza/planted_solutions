import numpy as np

import pickle

def load_planted_solution_tensor(moltag):
    filename = f'PlantedSolutionTensors/{moltag}_tensor'
    with open(filename, 'rb') as f:
        tensor = pickle.load(f)
    return tensor