                 #       ['Emitter 5 < -65']
        # ['Emitter 6 < -65',           'Emitter 0 < -44']
# ['Emitter 3 < -15', 'Emitter 0 < -45', 'Room 0', 'Emitter 0 < -44']
# ['Emitter 5 < -66', 'Emitter 0 < -35', 'Room 0', 'Emitter 0 < -45', None, None, 'Room 0', 'Emitter 0 < -44']
# ['Room 3', 'Room 2', 'Room 0', 'Room 2', None, None, 'Room 0', 'Room 2', None, None, None, None, None, None, 'Room 0', 'Room 2']

#https://stackoverflow.com/questions/47048228/converting-a-preorder-traversal-array-to-an-level-order-traversal-array-or-vice
import numpy as np

def level_to_pre(level_tree, index, preordered_tree):
    if index>=len(level_tree): 
        return preordered_tree #nodes at ind don't exist

    preordered_tree.append(level_tree[index]) #append to back of the array
    preordered_tree = level_to_pre(level_tree,index*2+1,preordered_tree) #recursive call to left
    preordered_tree = level_to_pre(level_tree,index*2+2,preordered_tree) #recursive call to right
    return preordered_tree

def convert_to_preorder_array(matrix):
    np_matrix = np.array([])
    for i in matrix:
        np.append(np.matrix, np.array(i))
    
    # np_matrix = np.array(matrix)
    print("MATRIX", np_matrix)
    # Multiplying arrays
    np_array = np_matrix.flatten()
    print("NP_ARRAY", np_array)
    array = np_array.tolist()
    
    return level_to_pre(array,0,[])
    
if __name__ == "__main__":
    print(convert_to_preorder_array([[1],[2,3],[4,5]]))