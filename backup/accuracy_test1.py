import numpy as np
from decision_tree import Classifier #own script
from prepare_data import prepare_data
from prepare_data_cross_val import prepare_data_cross_val
import evaluation
import sys

# if sys.argv[0] != None:
#     file_name = sys.argv[0]
# else:
#     file_name = 'noisy_dataset.txt'
# # file_path = r'intro2ML-coursework1/wifi_db/' + 'file_name'
# file_path = r'intro2ML-coursework1/wifi_db/clean_dataset.txt'
file_path = r'intro2ML-coursework1/wifi_db/noisy_dataset.txt'
full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

# full_dataset = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4]])
seed = 4
random_gen = np.random.default_rng(seed)

test_fold, val_fold, train_fold = prepare_data_cross_val(full_dataset, random_gen=random_gen)
# train_fold, val_fold, test_fold = crossval(full_dataset, random_gen=random_gen, outer_folds=4, inner_folds=3)
# print(test_fold[0][:-1])
print('test_fold:  ',np.shape(test_fold))
print('val_fold:   ',np.shape(val_fold))
print('train_fold: ',np.shape(train_fold))
# print(val_fold[0:2])
# print(train_fold[0:2])
# quit()
# train_fold, val_fold, test_fold = cross_val(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen)

val_sum = 0
sum_val_pruned = 0
sum_val_pruned2 = 0
sum_val_pruned_two = 0
test_sum = 0
sum_test_pruned = 0
sum_test_pruned2 = 0
sum_test_pruned_two = 0
sum_nodes_pruned = 0
sum_nodes_pruned2 = 0
sum_nodes = 0
# training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen, save_name='attempt1')
for index in range(len(train_fold)): 
    training_data = np.array(train_fold[index])
    validation_data = np.array(val_fold[index])
    test_data = np.array(test_fold[index%9])
    # print('some test data: ', test_data[:])
    # print('some test data[:,:-1]: ', test_data[:,:-1])
    tree = Classifier.fit(training_data)

    # ACCURACY OF OVERFITTED TREE ONTO VALIDATION DATA
    val_predictions = Classifier.predict(tree, validation_data[:,:-1])
    val_confusion_matrix = evaluation.calc_confusion_matrix(validation_data, val_predictions) 
    val_acc = evaluation.calc_accuracy(validation_data, val_predictions)
    
    print('--- FOLD ', index,' ---')
    print("Accuracy(original, validation):", val_acc)
    val_sum += val_acc

    # ACCURACY OF OVERFITTED TREE ONTO TEST DATA
    predictions = Classifier.predict(tree, test_data[:,:-1]) #test_data
    confusion_matrix = evaluation.calc_confusion_matrix(test_data, predictions) #test_data
    test_acc = evaluation.calc_accuracy(test_data, predictions) #test_data

    print("Accuracy(original, test):", test_acc)
    test_sum += test_acc


    precisions = evaluation.calc_precision(confusion_matrix)
    recalls = evaluation.calc_recall(confusion_matrix)
    f1s = evaluation.calc_F1(precisions, recalls)
    # for room in range(len(precisions)):
        # print(f"Precision for room {room+1}: {precisions[room]:.6f}")
        # print(f"Recall for room {room+1}: {recalls[room]:.6f}")
        # print(f"F1 for room {room+1}: {f1s[room]:.6f}")

    nodes, leaves, depth = tree.tree_properties(tree.root)
    # print("Tree with:")
    print(f"\t{nodes} nodes")
    # print(f"\t{leaves} leaves")
    print(f"\t{depth} depth")
    # tree.print_tree()


    # PRUNED TREE
    print('////')
    pruned_tree, depth = Classifier.prune(tree, validation_data)
    # VALIDATION ACCURACY SANITY CHECK
    val_predictions_pruned = Classifier.predict(pruned_tree, validation_data[:,:-1]) #test_data[:,:-1]
    val_confusion_matrix_pruned = evaluation.calc_confusion_matrix(validation_data, val_predictions_pruned) #test_data
    val_acc_pruned = evaluation.calc_accuracy(validation_data, val_predictions_pruned) #test_data
    print("Accuracy Pruned(validation):", val_acc_pruned)
    sum_val_pruned += val_acc_pruned

    # TEST DATA ON THE PRUNED TREE 
    test_predictions_pruned = Classifier.predict(pruned_tree, test_data[:,:-1]) #test_data[:,:-1]
    test_confusion_matrix_pruned = evaluation.calc_confusion_matrix(test_data, test_predictions_pruned) #test_data
    test_acc_pruned = evaluation.calc_accuracy(test_data, test_predictions_pruned) #test_data
    print("Accuracy Pruned(test):", test_acc_pruned)
    sum_test_pruned += test_acc_pruned

    nodes_pruned, leaves_pruned, depth_pruned = pruned_tree.tree_properties(pruned_tree.root)
    # print("Tree with:")
    print(f"\t{nodes_pruned} nodes")
    # print(f"\t{leaves_pruned} leaves")
    print(f"\t{depth_pruned} depth")
    # tree.print_tree()
    # print('clasifier acc: ', pruned_tree.current_accuracy)
    # diff = pruned_tree.current_accuracy - acc_pruned

    # PRUNED 2 TREE
    print('////')
    pruned_tree2, depth2 = Classifier.prune(pruned_tree, validation_data)
    # VALIDATION ACCURACY SANITY CHECK
    val_predictions_pruned2 = Classifier.predict(pruned_tree2, validation_data[:,:-1]) #test_data[:,:-1]
    val_confusion_matrix_pruned2 = evaluation.calc_confusion_matrix(validation_data, val_predictions_pruned2) #test_data
    val_acc_pruned2 = evaluation.calc_accuracy(validation_data, val_predictions_pruned2) #test_data
    print("Accuracy Pruned(validation):", val_acc_pruned2)
    sum_val_pruned2 += val_acc_pruned2

    # TEST DATA ON THE PRUNED TREE 
    test_predictions_pruned2 = Classifier.predict(pruned_tree2, test_data[:,:-1]) #test_data[:,:-1]
    test_confusion_matrix_pruned2 = evaluation.calc_confusion_matrix(test_data, test_predictions_pruned2) #test_data
    test_acc_pruned2 = evaluation.calc_accuracy(test_data, test_predictions_pruned2) #test_data
    print("Accuracy Pruned(test):", test_acc_pruned2)
    sum_test_pruned2 += test_acc_pruned2

    nodes_pruned2, leaves_pruned2, depth_pruned2 = pruned_tree2.tree_properties(pruned_tree2.root)
    # print("Tree with:")
    print(f"\t{nodes_pruned2} nodes")
    # print(f"\t{leaves_pruned} leaves")
    print(f"\t{depth_pruned2} depth")
    # tree.print_tree()
    # print('clasifier acc: ', pruned_tree.current_accuracy)
    # diff = pruned_tree.current_accuracy - acc_pruned

    diff_val = val_acc_pruned - val_acc
    print("Difference(validation): ", diff_val)
    diff_test = test_acc_pruned - test_acc
    print("Difference(test): ", diff_test)
    diff_nodes = nodes - nodes_pruned
    print("Difference(nodes): ", diff_nodes)
    sum_nodes_pruned += nodes_pruned
    sum_nodes_pruned2 += nodes_pruned2
    sum_nodes += nodes

    # SECOND ROUND OF PRUNING
    # print('////')
    # pruned_tree_two, depth = Classifier.prune(tree, validation_data)
    # # VALIDATION ACCURACY SANITY CHECK
    # val_predictions_pruned_two = Classifier.predict(pruned_tree_two, validation_data[:,:-1]) #test_data[:,:-1]
    # val_confusion_matrix_pruned_two = evaluation.calc_confusion_matrix(validation_data, val_predictions_pruned_two) #test_data
    # val_acc_pruned_two = evaluation.calc_accuracy(validation_data, val_predictions_pruned_two) #test_data
    # print("Accuracy Pruned(validation):", val_acc_pruned_two)
    # sum_val_pruned_two += val_acc_pruned_two

    # # TEST DATA ON THE PRUNED TREE 
    # test_predictions_pruned_two = Classifier.predict(pruned_tree_two, test_data[:,:-1]) #test_data[:,:-1]
    # test_confusion_matrix_pruned_two = evaluation.calc_confusion_matrix(test_data, test_predictions_pruned_two) #test_data
    # test_acc_pruned_two = evaluation.calc_accuracy(test_data, test_predictions_pruned_two) #test_data
    # print("Accuracy Pruned(test):", test_acc_pruned_two)
    # sum_test_pruned_two += test_acc_pruned_two

    # nodes, leaves, depth = pruned_tree_two.tree_properties(pruned_tree_two.root)
    # # print("Tree with:")
    # print(f"\t{nodes} nodes")
    # # print(f"\t{leaves} leaves")
    # print(f"\t{depth} depth")
        

# AVERAGES FOR VALIDATION DATA
print('dataset used: ',file_path)
avg_val = val_sum/len(train_fold)
avg_pruned_val = sum_val_pruned/len(train_fold)
diffs_val = avg_pruned_val - avg_val
print("Average(val): ", avg_val)
print("Average Pruned(val): ", avg_pruned_val)
print("Difference betweeen pruned and normal: ", diffs_val)

# AVERAGES FOR TEST DATA
print('dataset used: ',file_path)
avg_test = test_sum/len(train_fold)
avg_pruned_test = sum_test_pruned/len(train_fold)
diffs_test = avg_pruned_test - avg_test
print("Average(test): ", avg_test)
print("Average Pruned(test): ", avg_pruned_test)
print("Difference betweeen pruned and normal: ", diffs_test)

avg_nodes_pruned = sum_nodes_pruned/90
avg_nodes_pruned2 = sum_nodes_pruned2/90
avg_nodes = sum_nodes/90
print("Average nodes (normal tree):", avg_nodes)
print("Average nodes (pruned tree):", avg_nodes_pruned)
print("Average nodes (pruned tree):", avg_nodes_pruned2)

