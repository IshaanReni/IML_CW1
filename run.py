import numpy as np
from decision_tree import Classifier
from prepare_data import prepare_data 
import evaluation
import sys  #python standard library (allowed)

################# INPUT ##################

if len(sys.argv) > 1:
    file_name = sys.argv[1]     #first argument from terminal is filename
else:
    file_name = 'noisy_dataset.txt'

file_path = r'intro2ML-coursework1/wifi_db/' + file_name

full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

################# DATA PREPARATION ##################

seed = 4
random_gen = np.random.default_rng(seed)

training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=random_gen, save_name='attempt1') #prepare data (saving to txt optional)

################# TRAINING ##################

tree = Classifier.fit(training_data, max_depth=10)  #create tree trained on training data
predictions = Classifier.predict(tree, test_data[:,:-1])    #query tree on test data

################# STATISTICS ##################

confusion_matrix = evaluation.calc_confusion_matrix(test_data, predictions)
acc = evaluation.calc_accuracy(test_data, predictions)

print("Accuracy prepruned:", acc)
print('confusion matrix:')
print(confusion_matrix)

precisions = evaluation.calc_precision(confusion_matrix)
recalls = evaluation.calc_recall(confusion_matrix)
f1s = evaluation.calc_F1(precisions, recalls)

for room in range(len(precisions)):
    print(f"Precision for room {room+1}: {precisions[room]:.6f}")
    print(f"Recall for room {room+1}: {recalls[room]:.6f}")
    print(f"F1 for room {room+1}: {f1s[room]:.6f}")

nodes, leaves, depth = tree.tree_properties(tree.root)    #get properties
print("Tree with:")
print(f"\t{nodes} nodes")
print(f"\t{leaves} leaves")
print(f"\t{depth} depth")

tree.print_tree() #first tree printed (halts program until closed!)

################# PRUNING ##################

pruned_tree, depth = Classifier.prune(tree, validation_data)    #return a tree pruned on validation data
predictions_pruned = Classifier.predict(pruned_tree, test_data[:,:-1])    #query tree on test data

################# PRUNING STATISTICS ##################

confusion_matrix_pruned = evaluation.calc_confusion_matrix(validation_data, predictions_pruned)
print(confusion_matrix_pruned)
acc_pruned = evaluation.calc_accuracy(test_data, predictions_pruned)

print("Accuracy:", acc_pruned)
print('confusion matrix:')
print(confusion_matrix_pruned)

nodes_pruned, leaves_pruned, depth_pruned = pruned_tree.tree_properties(pruned_tree.root)   #get properties
print("Tree with:")
print(f"\t{nodes_pruned} nodes")
print(f"\t{leaves_pruned} leaves")
print(f"\t{depth_pruned} depth")

diff_test = acc_pruned - acc
print("Difference accuracy: ", diff_test)   #difference between pruned and unpruned
diff_nodes = nodes - nodes_pruned
print("Difference nodes: ", diff_nodes)     #difference between pruned and unpruned

pruned_tree.print_tree()    #pruned tree printed
