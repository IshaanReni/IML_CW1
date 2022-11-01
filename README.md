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
The files run.py and run_cross_val.py are examples of how to interface with the classes and functions in the library.

Load other datasets into `intro2ML-coursework1/wifi_db/`. All datasets must be in `.txt` format.

In order to run training, pruning, and evaluation once, use the following command in the IML_CW1 directory:
````shell
python3 run.py [dataset-filename]
````
The optional argument [dataset-filename] replaces the default .txt file with the specified .txt file (in `intro2ML-coursework1/wifi_db/` directory). The '.txt.' file extension must be mentioned.  
Alternatively, the default file_name variable in run.py can be changed.

Similarly, cross-validation can be performed using the command:
````shell
python3 run_cross_val.py [dataset-filename]
````

## Using functions
In the case that a new top-level script is to be made, the key functions are described below.

````shell
training_data, validation_data, test_data = prepare_data(full_dataset, test_prop=0.1, val_prop=0.1, random_gen=np.random.default_rng(8), save_name=None)
````