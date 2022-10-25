import numpy as np

def calc_confusion_matrix(test_data, predictions): #test data is NxK numpy array, predictions is Nx1 numpy array
    ground_truth = test_data[:,-1] # e.g. [1 2 2 2 4 3 2 1 2 3 2 2 1 4 3 2 1 1 1]
    confusion_matrix = np.zeros((4,4)) #[[0 0 0 0] [0 0 0 0] [0 0 0 0] [0 0 0 0]]
    for test in range(len(predictions)):
        confusion_matrix[ground_truth[test]-1,predictions[test]-1] += 1 #for each test, add 1 to the intersection between actual and prediction

    return(confusion_matrix)


def calc_precision(matrix):
    precisions = []
    for i in range(len(matrix[0,:])):
        tp = matrix[i,i]
        fp = np.sum(matrix[:, i]) - tp
        pre = tp / (tp+fp)
        precisions.append(pre)

    return precisions

def calc_recall(matrix):
    recalls = []
    for i in range(len(matrix[0,:])):
        tp = matrix[i, i]
        fn = np.sum(matrix[i, :]) - tp
        re = tp / (tp+fn)
        recalls.append(re)

    return recalls

def calc_F1(prec, rec):
    f1s = []
    for p in range(len(prec)):
        ans = (2 * prec[p] * rec[p])/(prec[p] + rec[p])
        f1s.append(ans)
    return f1s