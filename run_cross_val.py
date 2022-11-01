import numpy as np
from prepare_data_cross_val import prepare_data_cross_val
from cross_validation import cross_validation
import sys  #python standard library (allowed)

################# INPUT ##################

if len(sys.argv) > 1:
    file_name = sys.argv[1]     #first argument from terminal is filename
else:
    file_name = 'noisy_dataset.txt'

file_path = r'./intro2ML-coursework1/wifi_db/' + file_name

full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

################# DATA PREPARATION ##################

seed = 4
random_gen = np.random.default_rng(seed)

outer_folds = 10
inner_folds = 9
test_folds, val_folds, train_folds = prepare_data_cross_val(full_dataset, random_gen, outer_folds, inner_folds) #create data for every fold

print('test_folds:  ', np.shape(test_folds))
print('val_folds:   ', np.shape(val_folds))
print('train_folds: ', np.shape(train_folds))

################# CROSS VALIDATION & STATISTICS ##################

cross_validation(test_folds, val_folds, train_folds, outer_folds, inner_folds)  #run cross validation