import numpy as np
from decision_tree import Classifier #own script
from prepare_data import prepare_data

file_path = r'intro2ML-coursework1\wifi_db\clean_dataset.txt'
full_dataset = np.loadtxt(r'intro2ML-coursework1\wifi_db\noisy_dataset.txt').astype(np.int64)    #load data from text file into integer numpy array

seed = 4
random_gen = np.random.default_rng(seed)

training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.2, val_prop=0, random_gen=random_gen, save_name='attempt1')

tree = Classifier.fit(training_data, max_depth=10)

predictions = Classifier.predict(tree, test_data[:,:-1])

print(np.sum(predictions==test_data[:,-1]))

tree.print_tree()
