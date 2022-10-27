import numpy as np
from decision_tree import Classifier #own script
from prepare_data import prepare_data
from cross_val import cross_val
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

# full_dataset = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4]])
seed = 4
random_gen = np.random.default_rng(seed)

train_fold, val_fold, test_fold = cross_val(full_dataset, random_gen=random_gen)
# train_fold, val_fold, test_fold = crossval(full_dataset, random_gen=random_gen, outer_folds=4, inner_folds=3)
print(test_fold[0:3,:-1])
print(np.shape(test_fold))
# print(val_fold[0:2])
# print(train_fold[0:2])
# quit()
# train_fold, val_fold, test_fold = cross_val(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen)
sum = 0
sum_pruned = 0
# training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen, save_name='attempt1')
for index in range(len(train_fold)):
    training_data = train_fold[index]
    validation_data = val_fold[index]
    test_data = test_fold[index%9]
    print(test_data[0:3])

    tree = Classifier.fit(training_data)
    predictions = Classifier.predict(tree, test_data[:,:-1])

    confusion_matrix = evaluation.calc_confusion_matrix(test_data, predictions)
    acc = evaluation.calc_accuracy(test_data, predictions)

    print("Accuracy:", acc)
    sum += acc

    precisions = evaluation.calc_precision(confusion_matrix)
    recalls = evaluation.calc_recall(confusion_matrix)
    f1s = evaluation.calc_F1(precisions, recalls)
    # for room in range(len(precisions)):
        # print(f"Precision for room {room+1}: {precisions[room]:.6f}")
        # print(f"Recall for room {room+1}: {recalls[room]:.6f}")
        # print(f"F1 for room {room+1}: {f1s[room]:.6f}")

    # nodes, leaves, depth = tree.tree_properties(tree.root)
    # print("Tree with:")
    # print(f"\t{nodes} nodes")
    # print(f"\t{leaves} leaves")
    # print(f"\t{depth} depth")
    # tree.print_tree()

    Classifier.decision_tree_pruning(tree, tree.root, validation_data)
    predictions_pruned = Classifier.predict(tree, test_data[:,:-1])

    confusion_matrix_pruned = evaluation.calc_confusion_matrix(test_data, predictions_pruned)
    acc_pruned = evaluation.calc_accuracy(test_data, predictions_pruned)

    print("Accuracy Pruned:", acc_pruned)
    sum_pruned += acc_pruned
    # nodes, leaves, depth = tree.tree_properties(tree.root)
    # print("Tree with:")
    # print(f"\t{nodes} nodes")
    # print(f"\t{leaves} leaves")
    # print(f"\t{depth} depth")
    # tree.print_tree()

avg = sum/len(train_fold)
avg_pruned = sum_pruned/len(train_fold)
print("Average: ", avg)
print("Average Pruned: ", avg_pruned)

