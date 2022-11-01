import numpy as np

file_path = r'intro2ML-coursework1/wifi_db/clean_dataset.txt'
full_dataset = np.loadtxt(file_path).astype(np.int64)    #load data from text file into integer numpy array

for i, data in enumerate(full_dataset):###############
    for j, data2 in enumerate(full_dataset):
        if (data == data2).all():
            if i != j:
                print(data, data2, data == data2)##############