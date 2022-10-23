import numpy as np

#splits data into training, validation, and test
def prepare_data (full_dataset, test_prop=0.15, val_prop=0.05, random_gen=np.random.default_rng(8), save_name=None):
    shuffled_ind = random_gen.permutation(len(full_dataset))    #permute numbers between 0 and length of dataset
    test_amount = int(full_dataset.shape()[0]*test_prop)      #count of test data
    val_amount = int(full_dataset.shape()[0]*val_prop)        #count of validation data
    print(val_amount, test_amount) ######
    test_data = full_dataset[shuffled_ind[:test_amount]]        #from beginning up until test amount is test data
    val_data = full_dataset[shuffled_ind[test_amount:test_amount+val_amount]]      #from test to test+val is validation data
    train_data = full_dataset[shuffled_ind[test_amount+val_amount:]]     #everything after test and val is training data

    if save_name is not None:
        with open(f"{save_name}_train.txt", 'w') as train_f:
            train_f.write(train_data)
        with open(f"{save_name}_val.txt", 'w') as val_f:
            train_f.write(val_data)
        with open(f"{save_name}_test.txt", 'w') as test_f:
            train_f.write(test_data)
        print(f"Saved prepared data under prefix: {save_name}")

    return train_data, val_data, test_data