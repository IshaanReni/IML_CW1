import numpy as np

def prepare_data_cross_val(full_dataset, random_gen=np.random.default_rng(8), outer_folds=10, inner_folds=9):
    # shuffled_ind = random_gen.permutation(len(full_dataset))    #permute numbers between 0 and length of dataset
    # test_prop = 1/outer_folds
    # val_prop = 1/outer_folds
    # test_amount = int(full_dataset.shape[0]*test_prop)      #count of test data
    # val_amount = int(full_dataset.shape[0]*val_prop)        #count of validation data
    test_folds = []
    val_folds = []
    train_folds = []

    random_gen.shuffle(full_dataset)
    nr_rows_per_fold = int(len(full_dataset) / outer_folds)
    for i in range(outer_folds):
        test_data = full_dataset[i*nr_rows_per_fold : (i+1)*nr_rows_per_fold]
        test_folds.append(test_data.tolist())
        leftover_data = np.delete(full_dataset, np.s_[i*nr_rows_per_fold : (i+1)*nr_rows_per_fold], axis=0)
        nr_rows_per_group = int(len(leftover_data) / 9)
        # print('test: ',len(test_data),' with data: ', test_data[0:2])
        # print('fold: ',len(test_folds),' with data: ', test_folds[len(test_folds)-1][0:2])
        # print('leftover: ',len(leftover_data),' with data: ', leftover_data[0:2])
        # print('//')
        for j in range(inner_folds):
            validation_data = leftover_data[j*nr_rows_per_group : (j+1)*nr_rows_per_group]
            train_data = np.delete(leftover_data,np.s_[j*nr_rows_per_group : (j+1)*nr_rows_per_group], axis=0)
            val_folds.append(validation_data.tolist())
            train_folds.append(train_data.tolist())
            # print('validation: ',len(validation_data),' with data: ',validation_data[0:2])
            # print('train: ',len(train_data),' with data: ', train_data[0:2])
            # print('//')
        # print('------------------------')
    # test_folds.reshape(outer_folds,nr_rows_per_group)
    # print('//val_fold:   ',len(val_folds),' with data: '  , val_folds[len(val_folds)-1][0:2])
    # print('//test_fold:  ',len(test_folds),' with data: ' , test_folds[len(test_folds)-1][0:2])
    # print('//train_fold: ',len(train_folds),' with data: ', train_folds[len(train_folds)-1][0:2])
    # print('//val_fold:   ',len(val_folds),' with data: '  , val_folds)
    # print('//test_fold:  ',len(test_folds),' with data: ' , test_folds)
    # print('//train_fold: ',len(train_folds),' with data: ', train_folds)

    return test_folds, val_folds, train_folds


    
    # for i in range(outer_folds):
    #     fold_dataset = full_dataset[]
    #     for fold_in in range(inner_folds):
    #         fold_dataset = np.roll(full_dataset, fold_in*test_amount)  #shifts the rows downwards by a set amount. Kind of like a ring ruffer.
    #         val_data = fold_dataset[shuffled_ind[:val_amount]]      #from test to test+val is validation data
    #         train_data = fold_dataset[shuffled_ind[val_amount:]]     #everything after test and val is training data 
    #         test_folds.append(test_data)
    #         val_folds.append(val_data)
    #         train_folds.append(train_data)
    #     fold_dataset = full_dataset
    # return test_folds, val_folds, train_folds