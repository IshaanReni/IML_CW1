import numpy as np
from decision_tree import Classifier #own script
from prepare_data import prepare_data

full_dataset = r'intro2ML-coursework1\wifi_db\clean_dataset.txt'

seed = 4
random_gen = np.random.default_rng(seed)

training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.2, val_prop=0, random_gen=random_gen)

print(training_data.size()/full_dataset.size())

