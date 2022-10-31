import numpy as np

def prepare_data_cross_val(full_dataset, random_gen=np.random.default_rng(8), outer_folds=10, inner_folds=9):
    # full_dataset = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4]])#####
    test_prop = 1/outer_folds
    val_prop = 1/inner_folds
    test_amount = int(full_dataset.shape[0]*test_prop)      #count of test data
    test_folds = []
    val_folds = []
    train_folds = []
    full_dataset_copy = full_dataset.copy()
    random_gen.shuffle(full_dataset_copy, axis=0)            #randomly shuffle data
    print(full_dataset_copy)#####

    for fold in range(outer_folds):
        full_dataset_copy = np.roll(full_dataset_copy, test_amount, axis=0)  #shifts the rows downwards by a set amount. Similar to ring buffer.
        test_folds.append(full_dataset_copy[:test_amount])      #from test to test+val is validation data
        fold_dataset = full_dataset_copy[test_amount:].copy()           #everything after test is val and training data 
        val_amount = int(fold_dataset.shape[0]*val_prop)        #count of validation data
        for fold_in in range(inner_folds):
            fold_dataset = np.roll(fold_dataset, val_amount, axis=0)  #shifts the rows downwards by a set amount. Similar to ring buffer.
            val_folds.append(fold_dataset[:val_amount])      #from test to test+val is validation data
            train_folds.append(fold_dataset[val_amount:])     #everything after test and val is training data 
        
    #####RETURN METRICS
    return test_folds, val_folds, train_folds
