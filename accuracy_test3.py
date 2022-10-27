import numpy as np
from decision_tree import Classifier #own script
from prepare_data import prepare_data
from crossval import crossval
import evaluation
import sys

# if sys.argv[0] != None:
#     file_name = sys.argv[0]
# else:
#     file_name = 'noisy_dataset.txt'
# # file_path = r'intro2ML-coursework1/wifi_db/' + 'file_name'
file_path = r'intro2ML-coursework1/wifi_db/clean_dataset.txt'
# file_path = r'/IML_CW1/intro2ML-coursework1/wifi_db/noisy_dataset.txt'
full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

full_dataset = np.array([[0,0,0,0,0,0],[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4]])
seed = 4
random_gen = np.random.default_rng(seed)

# train_fold, val_fold, test_fold = cross_val(full_dataset, random_gen=random_gen)
test_fold, val_fold, train_fold = crossval(full_dataset, random_gen=random_gen, outer_folds=5, inner_folds=4)
print("test", test_fold)
print("val", val_fold)
print("train", train_fold)