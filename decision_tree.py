import numpy as np  #allowed


class Node:
#Tree node

    def __init__(self):
        self.true_child = self.false_child = None   #children initially pointing to None.


class Decision (Node):
#for non-leaves. Inherits Node class. Stores attribute and value of decision

    def __init__(self, attribute, value):
        super().__init__() #run inherited __init__() first
        self.attribute = attribute  #which wifi emitter to split by (column index)
        self.value = value  #value by which to split into two subsets, for a given attribute


class Leaf (Node):
#for leaves. Inherits Node class. Stores class label (room no.) for prediction
    
    def __init__(self, room):
        super().__init__() #run inherited __init__() first
        self.room = room   #final predicted class label


class DecisionTree:
#Tree stores root node

    #constructor for a tree
    def __init__(self):
        self.root = None    #beginning of tree
    
    #save model to text file after training
    def save_tree (self):
        pass

    #rebuild tree from saved model for predicting
    def load_tree (self, filepath):
        pass
    
    #This one doesnt actually work with our trees............
    # def print_tree(self, tree, lvl=1): ###### from online, replace with own implementation
    #     '''
    #     Print embedded lists as an indented text tree

    #     Recursive to depth 3

    #     tree: list[val, list[]...]
    #     '''
    #     indent = 2
    #     bullet = ''

    #     if lvl == 1: bullet = '*'
    #     elif lvl == 2: bullet = '+'
    #     elif lvl == 3: bullet = '-'
    #     else: raise ValueError('Unknown lvl: {}'.format(lvl))

    #     for i in tree:
    #         if type(i) is list:
    #             DecisionTree.print_tree(i, lvl+1)
    #         else:
    #             print('{}{} {}'.format(' '*(indent*(lvl-1)), bullet, i))


class Classifier:

    max_depth = None    #early stopping criteria for trees
    dataset = None  #full data array from file

    @classmethod
    def entropy(cls, x): ###### first first thing to be fixed!!!
        #x is a np.ndarray of size (N, K)
        values, count = np.unique(x, return_counts=True)
        entropy = 0

        for i in count:
            entropy += (-i / np.sum(count)) * np.log2(i / np.sum(count))

        return entropy

    @classmethod
    def entropy (dataset):  ####### check to see if this works and whether the data is divided like this
    # each col represents a different attribute
        col = np.size(dataset,1)
        entropy = 0
        for i in range(col-1):  #subtract 1 to ignore the last column which is the room label
            column = dataset[:,i]
            unique, count = np.unique(column, return_counts=True)   #check which is unique and count the number of unique val
            size = np.sum(count)  #total number of outcomes
            for j in count:
                prob = j / size 
                entropy += -1 * prob * np.log2(prob)
    
    return entropy
    
    @classmethod
    def information_gain(cls, x, y):
        pass
    
    @classmethod
    def find_split(cls, data): #####redo later
        attribute = np.random.randint(data.shape[1]-1) #random column ##########
        value = np.random.choice(data[:,attribute]) #random number from column #########
        return attribute, value

    #recursive function which constructs tree and returns subtree root node
    @classmethod
    def decision_tree_learning (cls, data, depth):
        if len(data) == 0: ###########temp for random split
            return Leaf(0), depth ###########temp for random split

        room_labels, label_counts = np.unique(data[:, 7], return_counts=True) #get room labels and frequencies present in current subset
        if len(room_labels) == 1 or depth == Classifier.max_depth:    #if all samples from the same room or max_depth reached (early stopping)
            room_plurality = room_labels[label_counts==max(label_counts)]   #predicted room is mode of room labels
            leaf_node = Leaf(room_plurality)    #create leaf node with this room prediction
            return leaf_node, depth     #return leaf node and current depth to parent node
        else:
            attribute, value = Classifier.find_split(data)    #find optimal attribute and value to split by for this subset
            decision_node = Decision(attribute, value)      #create new node based on split choices
            true_subset = data[data[:, attribute]<value]    #subset which follows the condition "attribute < value"
            false_subset = data[data[:, attribute]>=value]  #complement set (doesn't follow condition)
            decision_node.true_child, true_subtree_depth = Classifier.decision_tree_learning(true_subset, depth+1) #recursive call on true side of dataset
            decision_node.false_child, false_subtree_depth = Classifier.decision_tree_learning(false_subset, depth+1) #recursive call on false side of dataset
            return (decision_node, max(true_subtree_depth, false_subtree_depth)) #returns node and current max depth to parent node

    #callable by other functions to commence training
    @classmethod
    def fit (cls, dataset_filepath, max_depth):
        Classifier.max_depth = max_depth    #tree will stop constructing when this depth is reached
        Classifier.dataset = np.loadtxt(dataset_filepath).astype(np.int64)    #load data from text file into integer numpy array
        decision_tree = DecisionTree()     #instantiate blank tree
        decision_tree.root = Classifier.decision_tree_learning(Classifier.dataset, 1)  #start recursive training process (beginning with depth 1)
        return decision_tree

    @classmethod
    def predict (cls, load=False, tree_filepath=None): ###### Next one to do
        pass


#default main when file ran individually
if __name__ == "__main__":
    tree = Classifier.fit(r'intro2ML-coursework1\wifi_db\noisy_dataset.txt', 10)
    tree.print_tree(tree.root, 3)
