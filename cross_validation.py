import numpy as np
from decision_tree import Classifier # own script
import evaluation

def cross_validation (test_folds, val_folds, train_folds, outer_folds=10, inner_folds=9):

    val_sum = 0
    sum_val_pruned = 0
    test_sum = 0
    sum_test_pruned = 0
    sum_nodes_pruned = 0
    sum_nodes = 0

    sumPrec = [0,0,0,0]
    sumRec = [0,0,0,0]
    sumF1 = [0,0,0,0]

    sumPrecP = [0,0,0,0]
    sumRecP = [0,0,0,0]
    sumF1P = [0,0,0,0]

    for index in range(len(train_folds)):                           #completing cross validation, train folds and val_folds are the same size of outer_folds x inner_folds = 90
        training_data = np.array(train_folds[index])                #get the training dataset
        validation_data = np.array(val_folds[index])                #get the validation dataset
        test_data = np.array(test_folds[int(index/inner_folds)])    #get the specific test dataset specific to the train and validation dataset

        tree = Classifier.fit(training_data)                        #generate the model by fitting the data
        # print('test_data:  ',np.shape(test_data))                 #checks for expected shape (number of rows per outter_fold, (emmiters+label=8))
        # print('validation_data:   ',np.shape(validation_data))
        # print('training_data: ',np.shape(training_data))

        # # CHECK FOR REPEATED DATA POINTS IN THE CROSS VALIDATION SPLITS FOR EACH FOLD
        # for train in training_data:                               # checks for possible repeated rows between train and validation
        #     for val in validation_data:
        #         if (train == val).all():
        #             print('------------------------------------', val, train, train==val)
        # for val in validation_data:                                 # checks for possible repeated rows between test and validation
        #     for test in test_data:
        #         if (val == test).all():
        #             print('------------------------------------', test, val, val==test)


        # # ACCURACY OF NORMAL TREE ONTO VALIDATION DATA
        # val_predictions = Classifier.predict(tree, validation_data[:,:-1])
        # val_confusion_matrix = evaluation.calc_confusion_matrix(validation_data, val_predictions) 
        # val_acc = evaluation.calc_accuracy(validation_data, val_predictions)
        
        print('--- FOLD ', index,' ---')
        # print("Accuracy(original, validation):", val_acc)
        # val_sum += val_acc

        # ACCURACY OF NORMAL TREE ONTO TEST DATA
        predictions = Classifier.predict(tree, test_data[:,:-1])                        # removing the labels from every row in the data and running the predict function on them
        confusion_matrix = evaluation.calc_confusion_matrix(test_data, predictions)     # generating the confusion matrix comparing the real labels with the predictions
        test_acc = evaluation.calc_accuracy(test_data, predictions)                     # calculating accuracy based on real labels and prdictions

        print("Accuracy(original, test):", test_acc)
        test_sum += test_acc


        precisions = evaluation.calc_precision(confusion_matrix)                        # calculating precision for specific fold
        recalls = evaluation.calc_recall(confusion_matrix)                              # calculating recall for specific fold
        f1s = evaluation.calc_F1(precisions, recalls)                                   # calculating F1 for specific fold
        # for room in range(len(precisions)):                                           # evaluating metrics for each room
            # print(f"Precision for room {room+1}: {precisions[room]:.6f}")
            # print(f"Recall for room {room+1}: {recalls[room]:.6f}")
            # print(f"F1 for room {room+1}: {f1s[room]:.6f}")

        for room in range(len(precisions)):
            sumPrec[room] = sumPrec[room] + precisions[room]
            sumRec[room] = sumRec[room] + recalls[room]
            sumF1[room] = sumF1[room] + f1s[room]

        nodes, leaves, depth = tree.tree_properties(tree.root)                          # evaluating tree shape for each fold
        # print("Tree with:")
        print(f"\t{nodes} nodes")
        # print(f"\t{leaves} leaves")
        print(f"\t{depth} depth")

        # # # # # # PRUNE TREE generation for each fold
        print('//--------//')
        pruned_tree, depth = Classifier.prune(tree, validation_data)

        # TEST DATA ON THE PRUNED TREE 
        test_predictions_pruned = Classifier.predict(pruned_tree, test_data[:,:-1])                             # removing the labels from every row in the data and running the predict function on them
        test_confusion_matrix_pruned = evaluation.calc_confusion_matrix(test_data, test_predictions_pruned)     # generating the confusion matrix comparing the real labels with the predictions
        test_acc_pruned = evaluation.calc_accuracy(test_data, test_predictions_pruned)                          # calculating accuracy based on real labels and prdictions
        print("Accuracy Pruned(test):", test_acc_pruned)
        sum_test_pruned += test_acc_pruned 

        precisionsP = evaluation.calc_precision(test_confusion_matrix_pruned)
        recallsP = evaluation.calc_recall(test_confusion_matrix_pruned)
        f1sP = evaluation.calc_F1(precisionsP, recallsP)

        for roomP in range(len(precisionsP)):
            sumPrecP[roomP] = sumPrecP[roomP] + precisionsP[roomP]
            sumRecP[roomP] = sumRecP[roomP] + recallsP[roomP]
            sumF1P[roomP] = sumF1P[roomP] + f1sP[roomP]

        nodes_pruned, leaves_pruned, depth_pruned = pruned_tree.tree_properties(pruned_tree.root)               # evaluating tree shape for each fold
        print("Tree with:")
        print(f"\t{nodes_pruned} nodes")
        print(f"\t{leaves_pruned} leaves")
        print(f"\t{depth_pruned} depth")

        diff_test = test_acc_pruned - test_acc          # evaluating difference in accuracy between pruned and normal trees, per fold
        print("Difference(test): ", diff_test)
        diff_nodes = nodes - nodes_pruned               # evaluating difference in nodes between pruned and normal trees, per fold
        print("Difference(nodes): ", diff_nodes)
        sum_nodes_pruned += nodes_pruned
        sum_nodes += nodes

    # AVERAGES FOR TEST DATA
    avg_test = test_sum/len(train_folds)                    #calculating average accuracy for normal tree for whole cross validation
    avg_pruned_test = sum_test_pruned/len(train_folds)      #calculating average accuracy for pruned tree for whole cross validation
    diffs_test = avg_pruned_test - avg_test                 #calculating differences between average accuracies
    print("Average accuracy (test): ", avg_test)
    print("Average accuracy Pruned (test): ", avg_pruned_test)
    print("Difference betweeen pruned and normal: ", diffs_test)

    # AVERAGES FOR PRECISION DATA
    for i in range(len(sumPrec)):
        avgPrec = (sumPrec[i])/len(train_folds)
        avgPrecP = (sumPrecP[i])/len(train_folds)
        print(f'Avg Precision for Room {i+1}: {avgPrec:.6f}')
        print(f'Avg Pruned Precision for Room {i+1}: {avgPrecP:.6f}')
        avgRec = (sumRec[i])/len(train_folds)
        avgRecP = (sumRecP[i])/len(train_folds)
        print(f'Avg Recall for Room {i+1}: {avgRec:.6f}')
        print(f'Avg Pruned Recall for Room {i+1}: {avgRecP:.6f}')
        avgF1 = (sumF1[i])/len(train_folds)
        avgF1P = (sumF1P[i])/len(train_folds)
        print(f'Avg F1 for Room {i+1}: {avgF1:.6f}')
        print(f'Avg Pruned F1 for Room{i+1}: {avgF1P:.6f}')

    avg_nodes = sum_nodes/len(train_folds)                  #calculating average of number of nodes for normal tree (len(train_folds) = inner_folds x outer_folds)
    avg_nodes_pruned = sum_nodes_pruned/len(train_folds)    #calculating average of number of nodes for pruned tree
    print("Average nodes (normal tree):", avg_nodes)    
    print("Average nodes (pruned tree):", avg_nodes_pruned)

    tree.print_tree()               #   visualising normal tree
    pruned_tree.print_tree()        #   visualising pruned tree