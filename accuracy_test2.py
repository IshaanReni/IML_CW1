import numpy as np
from decision_tree import Classifier #own script
from prepare_data import prepare_data
from cross_val import cross_val
import evaluation
import sys

print(sys.argv)
if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = 'noisy_dataset.txt'
    # file_name = 'testDumb.txt'
file_path = r'intro2ML-coursework1/wifi_db/' + str(file_name)
# file_path = r'intro2ML-coursework1/wifi_db/clean_dataset.txt'
# file_path = r'/IML_CW1/intro2ML-coursework1/wifi_db/noisy_dataset.txt'
full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

# full_dataset = np.array([[1,1,1,1,1,1,1,1, 1],[2,2,2,2,2,2,2, 1],[3,3,3,3,3,3,3, 1],[4,4,4,4,4,4,4, 4]])
# print(np.shape(full_dataset))
seed = 4
random_gen = np.random.default_rng(seed)

#train_fold, val_fold, test_fold = cross_val(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen)

training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen, save_name='attempt1')

tree = Classifier.fit(training_data, 10)
predictions = Classifier.predict(tree, validation_data[:,:-1])

# print("first 30 predictions: \n", predictions[:30])
# print("first 30 ground truth: \n", test_data[:30,-1])

confusion_matrix = evaluation.calc_confusion_matrix(validation_data, predictions)
print(confusion_matrix)
acc = evaluation.calc_accuracy(validation_data, predictions)

print("Accuracy:", acc)
print('confusion matrix:')
print(confusion_matrix)

precisions = evaluation.calc_precision(confusion_matrix)
recalls = evaluation.calc_recall(confusion_matrix)
f1s = evaluation.calc_F1(precisions, recalls)
# for room in range(len(precisions)):
#     print(f"Precision for room {room+1}: {precisions[room]:.6f}")
#     print(f"Recall for room {room+1}: {recalls[room]:.6f}")
#     print(f"F1 for room {room+1}: {f1s[room]:.6f}")

nodes, leaves, depth = tree.tree_properties(tree.root)
print("Tree with:")
print(f"\t{nodes} nodes")
print(f"\t{leaves} leaves")
print(f"\t{depth} depth")
tree.print_tree()

Classifier.decision_tree_pruning(tree, tree.root, validation_data)

predictions_pruned = Classifier.predict(tree, validation_data[:,:-1]) #test_data

# print("first 30 predictions: \n", predictions[:30])
# print("first 30 ground truth: \n", test_data[:30,-1])

confusion_matrix_pruned = evaluation.calc_confusion_matrix(validation_data, predictions_pruned)
print(confusion_matrix_pruned)
# acc_pruned = evaluation.calc_accuracy(test_data, predictions_pruned)
acc_pruned = evaluation.calc_accuracy(validation_data, predictions_pruned)

print("Accuracy:", acc_pruned)
print('confusion matrix:')
print(confusion_matrix_pruned)


nodes, leaves, depth = tree.tree_properties(tree.root)
print("Tree with:")
print(f"\t{nodes} nodes")
print(f"\t{leaves} leaves")
print(f"\t{depth} depth")
tree.print_tree()