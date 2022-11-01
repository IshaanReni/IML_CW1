import numpy as np

def calc_confusion_matrix(test_data, predictions): # test data is NxK numpy array, predictions is Nx1 numpy array
    ground_truth = test_data[:,-1]      # e.g. [1 2 2 2 4 3 2 1 2 3 2 2 1 4 3 2 1 1 1]. test data used instead of predictions to avoid misclassification error in matrix shape
    confusion_matrix = np.zeros((4,4))  #[[0 0 0 0] [0 0 0 0] [0 0 0 0] [0 0 0 0]]
    for test in range(len(predictions)):
        confusion_matrix[ground_truth[test]-1,predictions[test]-1] += 1 # for each test, add 1 to the intersection between actual and prediction

    return confusion_matrix

def calc_accuracy(test_data, predictions):
    return np.sum(predictions==test_data[:,-1])/len(predictions)    # sum all values where the prediction equals the real label and divide by number of predicitons

def calc_precision(matrix):
    return np.diag(matrix)/np.sum(matrix,axis=0)                    # get an array of elements in the diagonal of the matrix and divide them by sum accross specific row

def calc_recall(matrix):
    return np.diag(matrix)/np.sum(matrix,axis=1)                    # get an array of elements in the diagolan of the matrix and divide them by sum accross specific collumn

def calc_F1(prec, rec):
    f1s = []
    for p in range(len(prec)):
        ans = (2 * prec[p] * rec[p])/(prec[p] + rec[p])             # applying F1 formula for every (precision,recall) tuples
        f1s.append(ans)
    return f1s