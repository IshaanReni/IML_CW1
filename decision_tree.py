import numpy as np
import matplotlib as plt

#Tree node (leaves are not nodes, but None singleton)
class Decision:

    def __init__(self, parent, attribute, value):   #OOP structure allows flexibility
        self.attribute = attribute  #which wifi emitter to split by
        self.value = value  #value by which to split into two subsets (for a given attribute)
        self.true = self.false = None   #children initially pointing to None
        self.parent = parent    #for reverse traversal
    
    def has_leaves(self): # Getter method to check if the decision has leaves (Clarification: this checks if the current node has children but no grand-children)######
        return not (self.true and self.false)

#Decision Tree Creation 
class DecisionTree:

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None    #beginning of tree
    
    def save_tree (self):
        pass

    def load_tree (self, filepath):
        pass

    @classmethod
    def entropy(x): ###### first first thing to be fixed!!!
        #x is a np.ndarray of size (N, K)
        values, count = np.unique(x, return_counts=True)
        entropy = 0

        for i in count:
            entropy += (-i / np.sum(count)) * np.log2(i / np.sum(count))

        return entropy
    
    @classmethod
    def information_gain(x, y):
        pass

# def find_split(data):
    
#     return 

def decision_tree_learning(data, depth):
    #if 
    pass
    

#callable by other functions to commence training
def fit (filepath, max_depth):
    file_data = np.loadtxt(filepath)
    decision_tree = DecisionTree(max_depth)

def predict (load=False, tree_filepath=None): ###### Next one to do
    pass


def print_tree(tree, lvl=1): ######
    '''
    Print embedded lists as an indented text tree

    Recursive to depth 3

    tree: list[val, list[]...]
    '''
    indent = 2
    bullet = ''

    if lvl == 1: bullet = '*'
    elif lvl == 2: bullet = '+'
    elif lvl == 3: bullet = '-'
    else: raise ValueError('Unknown lvl: {}'.format(lvl))

    for i in tree:
        if type(i) is list:
            print_tree(i, lvl+1)
        else:
            print('{}{} {}'.format(' '*(indent*(lvl-1)), bullet, i))

#default main when file ran individually
if __name__ == "__main__":
    max_depth_default = 10     #tree will stop constructing when this depth is reached
    #filepath = r'intro2ML-coursework1\wifi_db\clean_dataset.txt'
    filepath = r'intro2ML-coursework1\wifi_db\noisy_dataset.txt'
    dataset = np.loadtxt(filepath, dtype=int)   #integer array from text file

    #print(dataset)
