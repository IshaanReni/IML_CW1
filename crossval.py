import numpy as np
import evaluation
from decision_tree import Classifier

def crossval(full_dataset, random_gen=np.random.default_rng(8), outer_folds=10, inner_folds=9):
    # shuffled_ind = random_gen.permutation(len(full_dataset))    #permute numbers between 0 and length of dataset
    test_prop = 1/outer_folds
    val_prop = 1/inner_folds
    test_amount = int(full_dataset.shape[0]*test_prop)      #count of test data
    print("test amount", test_amount)
    val_amount = int(full_dataset.shape[0]*val_prop)        #count of validation data
    shuffled_ind = [i for i in range(len(full_dataset))]
    #shuffled_ind_small = [i for i in range(len(full_dataset)-test_amount)]
    #print(shuffled_ind_small)
    test_folds = []
    val_folds = []
    train_folds = []
    
    # for i in range(outer_folds):
    #     steps = i * test_amount
    #     test_data = full_dataset[shuffled_ind[steps:test_amount+steps]]
    #     test_folds.append(test_data)
    full_dataset_copy = full_dataset.copy()
    for fold in range(outer_folds):
        #print(full_dataset_copy)
        test_data = full_dataset_copy[shuffled_ind[:test_amount]]      #from test to test+val is validation data
        fold_dataset = full_dataset_copy[shuffled_ind[test_amount:]]     #everything after test and val is training data 
        test_folds.append(test_data)
        full_dataset_copy = np.roll(full_dataset_copy, test_amount, axis=0)  #shifts the rows downwards by a set amount. Kind of like a ring ruffer.
        for fold_in in range(inner_folds):
            # print(fold_dataset)
            val_data = fold_dataset[shuffled_ind[:val_amount]]      #from test to test+val is validation data
            train_data = fold_dataset[shuffled_ind[val_amount:]]     #everything after test and val is training data 
            ####RUN CROSSVAL HERE
            val_folds.append(val_data)
            train_folds.append(train_data)
            fold_dataset = np.roll(fold_dataset, val_amount, axis=0)  #shifts the rows downwards by a set amount. Kind of like a ring ruffer.
            # print("shuffle", shuffled_ind_small)
            # print("test_data", test_data)
            # print("val_data", val_data)
            # print("train_data", train_data)
        
        tree = Classifier.fit(train_data)    #Make the tree using the training data
        predictions = Classifier.predict(tree, test_data[:,:-1])    #Make the predictions
        confusion_matrix = evaluation.calc_confusion_matrix(test_data[fold], predictions)
        print(confusion_matrix)
        acc = evaluation.calc_accuracy(test_data, predictions)
        print("Accuracy:", acc)
        print('confusion matrix:')
        print(confusion_matrix)

        precisions = evaluation.calc_precision(confusion_matrix)
        recalls = evaluation.calc_recall(confusion_matrix)
        f1s = evaluation.calc_F1(precisions, recalls)
        
        nodes, leaves, depth = tree.tree_properties(tree.root)
        print("Tree with:")
        print(f"\t{nodes} nodes")
        print(f"\t{leaves} leaves")
        print(f"\t{depth} depth")
        tree.print_tree()

        Classifier.decision_tree_pruning(tree, tree.root, validation_data)

        predictions_pruned = Classifier.predict(tree, test_data[:,:-1])
        confusion_matrix_pruned = evaluation.calc_confusion_matrix(test_data, predictions_pruned)
        #print(confusion_matrix_pruned)
        acc_pruned = evaluation.calc_accuracy(test_data, predictions_pruned)

        print("Accuracy:", acc_pruned)
        print('confusion matrix:')
        print(confusion_matrix_pruned)

        nodes, leaves, depth = tree.tree_properties(tree.root)
        print("Tree with:")
        print(f"\t{nodes} nodes")
        print(f"\t{leaves} leaves")
        print(f"\t{depth} depth")
        tree.print_tree()   



    #####RETURN METRICS
    return np.array(test_folds), np.array(val_folds), np.array(train_folds)
