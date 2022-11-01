# IML_CW1
DoC: Introduction to Machine Learning  
Decision Tree Coursework

## Background
This library supports the training, pruning, and evaluation of decision trees, as well as cross-validation.
The data is assumed to be in numpy integer format, with the last column representing class labels.

## Setup
Clone the repo onto the lab machine using the following command.
````shell
git clone https://github.com/IshaanReni/IML_CW1.git
````
  
## Usage
The files `run.py` and `run_cross_val.py` are examples of how to interface with the classes and functions in the library.
  
Load other datasets into `intro2ML-coursework1/wifi_db/`. All datasets must be in `.txt` format.
  
In order to run training, pruning, and evaluation once, use the following command in the IML_CW1 directory:
````shell
python3 run.py [dataset-filename]
````
The optional argument [dataset-filename] replaces the default `.txt` file with the specified `.txt` file (in `intro2ML-coursework1/wifi_db/` directory). The `.txt.` file extension must be mentioned.  
Alternatively, the default file_name variable in `run.py` can be changed.
  
Similarly, cross-validation can be performed using the command:
````shell
python3 run_cross_val.py [dataset-filename]
````
  
## Using functions
In the case that a new top-level script is to be made, the key functions are described below.
(The library stores nodes and trees as objects, as OOP is used. `Classifier` denotes the class in which the class method resides.)
  
````python
from prepare_data import prepare_data 
training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=np.random.default_rng(8), save_name=None)
````
This function splits the full dataset (numpy array) into usable training, validation, and test data numpy arrays, in the proportions specified. The random generator can be overwritten if a different seed is desired.  
If the `save_name` variable is a string, the function will save the output arrays as txt files in `intro2ML-coursework1/wifi_db/`.
  
````python
from decision_tree import Classifier
tree = Classifier.fit(training_data, max_depth=999)
````
Returns a tree trained on the input training data (numpy array). Optional maximum depth can be specified for early stopping regularisation, debugging, or simpler graphs.
  
````python
from decision_tree import Classifier
pruned_tree, depth = Classifier.prune(tree, validation_data)
````
Returns a copy of the input tree which is pruned using validation data, along with its depth (tuple).
  
````python
from decision_tree import Classifier
predictions = Classifier.predict(tree, test_data[:,:-1])
````
Returns an array of predictions by the input tree, for each test in a test set. The function will work regardless of if the test class labels are sent into the function.
  
````python
import evaluation
confusion_matrix = evaluation.calc_confusion_matrix(test_data, predictions)
accuracy = evaluation.calc_accuracy(test_data, predictions)
precisions = evaluation.calc_precision(confusion_matrix)
recalls = evaluation.calc_recall(confusion_matrix)
f1s = evaluation.calc_F1(precisions, recalls)
````
`calc_confusion_matrix` returns an N×N confusion matrix, where N is the number of unique classes.  
The accuracy function doesn't use the confusion matrix to preclude the matrix needing to be made every fold in cross-validation (optimisation).  
Other evaluation functions use the matrix to return a per-class list of values.  

````python
from decision_tree import Classifier
nodes, leaves, depth = tree.tree_properties(tree.root)
nodes = tree.get_nodes()
leaves = tree.get_leaves()
depth = tree.get_depth()
````
Getter functions for obtaining tree statistics

````python
from prepare_data_cross_val import prepare_data_cross_val
test_folds, val_folds, train_folds = prepare_data_cross_val(full_dataset, random_gen=np.random.default_rng(8), outer_folds=10, inner_folds=9)
````
Creates lists of data to be used in folds. `test_folds` is `outer_folds` long whereas `val_folds` and `train_folds` is `outer_folds`×`inner_folds` long.  
Defaults to 10:10:80 split (90 folds in total).

````python
from cross_validation import cross_validation
cross_validation(test_folds, val_folds, train_folds, outer_folds, inner_folds)
````
Runs `outer_folds`×`inner_folds` cross-validation on the data. Prints metrics every fold (pre- and post-pruning), and gives overall metrics at the end.
  
````python
from decision_tree import Classifier
tree.print_tree()
````
Creates a `matplotlib` visualisation of the tree. Usually run last, as it halts the program while the graph window is open.
![image](https://user-images.githubusercontent.com/93332879/199309005-5ded8375-693f-44b1-bebe-da41d59d1c38.png)
