import numpy as np

def prepare_data_cross_val(full_dataset, random_gen=np.random.default_rng(8), outer_folds=10, inner_folds=9): 
    # preparing the data to be split in folds for cross validation
    # by default we have outer folds of 10 and inner folds of 9, but this can be overwritten
    test_folds = []
    val_folds = []
    train_folds = []

    random_gen.shuffle(full_dataset)                                                                        #shuffle dataset
    nr_rows_per_fold = int(len(full_dataset) / outer_folds)                                                 #get number of rows per outer fold
    for i in range(outer_folds):                                                                            #loop over the outer folds 
        test_data = full_dataset[i*nr_rows_per_fold : (i+1)*nr_rows_per_fold]                               #set one of the folds to be the test data
        test_folds.append(test_data.tolist())                                                               #append gotten test datat array onto the cross validation test folds
        leftover_data = np.delete(full_dataset, np.s_[i*nr_rows_per_fold : (i+1)*nr_rows_per_fold], axis=0) #take out the test data rows from the full datatset to assign validation and training data sets
        nr_rows_per_group = int(len(leftover_data) / inner_folds)                                           #get number of rows per inner fold
        for j in range(inner_folds):                                                                        #loop through inner folds
            validation_data = leftover_data[j*nr_rows_per_group : (j+1)*nr_rows_per_group]                  #assign one of the inner folds as validation dataset
            train_data = np.delete(leftover_data,np.s_[j*nr_rows_per_group : (j+1)*nr_rows_per_group], axis=0)  #the rest of the folds will be assigned as training data
            val_folds.append(validation_data.tolist())                                                      #append the data onto the vallidation folds array
            train_folds.append(train_data.tolist())                                                         #append data onto the training folds array

    # print('//val_fold:   ',len(val_folds),' with data: '  , val_folds[len(val_folds)-1][0:2])
    # print('//test_fold:  ',len(test_folds),' with data: ' , test_folds[len(test_folds)-1][0:2])
    # print('//train_fold: ',len(train_folds),' with data: ', train_folds[len(train_folds)-1][0:2])
    # print('//val_fold:   ',len(val_folds),' with data: '  , val_folds)
    # print('//test_fold:  ',len(test_folds),' with data: ' , test_folds)
    # print('//train_fold: ',len(train_folds),' with data: ', train_folds)

    return test_folds, val_folds, train_folds