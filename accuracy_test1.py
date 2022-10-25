import numpy as np
from decision_tree import Classifier #own script
from prepare_data import prepare_data
import evaluation

file_path = r'intro2ML-coursework1/wifi_db/noisy_dataset.txt'
full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

seed = 4
random_gen = np.random.default_rng(seed)
training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0, random_gen=random_gen, save_name='attempt1')

tree = Classifier.fit(training_data, 7)
predictions = Classifier.predict(tree, test_data[:,:-1])

print("first 30 predictions: \n", predictions[:30])
print("first 30 ground truth: \n", test_data[:30,-1])

confusion_matrix = evaluation.calc_confusion_matrix(test_data, predictions)

print("Accuracy:", np.sum(predictions==test_data[:,-1])/len(predictions))
print('confusion matrix:')
print(confusion_matrix)


precisions = evaluation.calc_precision(confusion_matrix)
recalls = evaluation.calc_recall(confusion_matrix)
f1s = evaluation.calc_F1(precisions, recalls)
for room in range(len(precisions)):
    print(f"Precision for room {room+1}: {precisions[room]:.6f}")
    print(f"Recall for room {room+1}: {recalls[room]:.6f}")
    print(f"F1 for room {room+1}: {f1s[room]:.6f}")

tree.print_tree()

