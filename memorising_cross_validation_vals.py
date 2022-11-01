import numpy as np
from decision_tree import Classifier #own script
from prepare_data_cross_val import prepare_data_cross_val
import evaluation

# # # SAME FUNCTIONALITY AS THE FILE AS cross_validation.py but stores the outputs of the terminal into different txt files

# file_path = r'intro2ML-coursework1/wifi_db/clean_dataset.txt'
file_path = r'intro2ML-coursework1/wifi_db/noisy_dataset.txt'
full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

seed = 4
random_gen = np.random.default_rng(seed)
outer_folds = 10
inner_folds = 9
test_folds, val_folds, train_folds = prepare_data_cross_val(full_dataset, random_gen, outer_folds, inner_folds)

print('test_folds:  ',np.shape(test_folds))
print('val_folds:   ',np.shape(val_folds))
print('train_folds: ',np.shape(train_folds))


val_sum = 0
sum_val_pruned = 0
test_sum = 0
sum_test_pruned = 0
sum_nodes_pruned = 0
sum_nodes = 0

with open('scatter_normal_nodes.txt', 'w') as nnfile:
    with open('scatter_normal_leaves.txt', 'w') as nlfile:
        with open('scatter_prune_nodes.txt', 'w') as pnfile:
            with open('scatter_prune_leaves.txt', 'w') as plfile:
                # training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen, save_name='attempt1')
                for index in range(len(train_folds)): 
                    training_data = np.array(train_folds[index])
                    validation_data = np.array(val_folds[index])
                    test_data = np.array(test_folds[int(index/inner_folds)])
                    # print('some test data: ', test_data[:])
                    # print('some test data[:,:-1]: ', test_data[:,:-1])
                    tree = Classifier.fit(training_data)
                    print('test_data:  ',np.shape(test_data))
                    print('validation_data:   ',np.shape(validation_data))
                    print('training_data: ',np.shape(training_data))

                    # for train in training_data:###############
                    #     for val in validation_data:
                    #         if (train == val).all():
                    #             print('------------------------------------', val, train, train==val)##############

                    for val in validation_data:###############
                        for test in test_data:
                            if (val == test).all():
                                print('------------------------------------', test, val, val==test)##############


                    # # ACCURACY OF OVERFITTED TREE ONTO VALIDATION DATA
                    # val_predictions = Classifier.predict(tree, validation_data[:,:-1])
                    # val_confusion_matrix = evaluation.calc_confusion_matrix(validation_data, val_predictions) 
                    # val_acc = evaluation.calc_accuracy(validation_data, val_predictions)
                    
                    print('--- FOLD ', index,' ---')
                    # print("Accuracy(original, validation):", val_acc)
                    # val_sum += val_acc

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

                    # PRUNE TREE
                    print('////')
                    pruned_tree, depth = Classifier.prune(tree, validation_data)

                    # TEST DATA ON THE PRUNED TREE 
                    test_predictions_pruned = Classifier.predict(pruned_tree, test_data[:,:-1])
                    test_confusion_matrix_pruned = evaluation.calc_confusion_matrix(test_data, test_predictions_pruned)
                    test_acc_pruned = evaluation.calc_accuracy(test_data, test_predictions_pruned)
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

                    diff_test = test_acc_pruned - test_acc
                    print("Difference(test): ", diff_test)
                    diff_nodes = nodes - nodes_pruned
                    print("Difference(nodes): ", diff_nodes)
                    sum_nodes_pruned += nodes_pruned
                    sum_nodes += nodes
                    nnfile.write(str(nodes) + "\n")
                    nlfile.write(str(leaves) + "\n")
                    pnfile.write(str(nodes_pruned) + "\n")
                    plfile.write(str(leaves_pruned) + "\n")

# AVERAGES FOR TEST DATA
print('dataset used: ',file_path)
avg_test = test_sum/len(train_folds)
avg_pruned_test = sum_test_pruned/len(train_folds)
diffs_test = avg_pruned_test - avg_test
print("Average(test): ", avg_test)
print("Average Pruned(test): ", avg_pruned_test)
print("Difference betweeen pruned and normal: ", diffs_test)

avg_nodes_pruned = sum_nodes_pruned/90
avg_nodes = sum_nodes/90
print("Average nodes (normal tree):", avg_nodes)
print("Average nodes (pruned tree):", avg_nodes_pruned)

tree.print_tree()
pruned_tree.print_tree()
