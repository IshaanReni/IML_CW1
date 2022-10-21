import numpy as np  #allowed
from binaryTreeVisualise import plot_tree #own file
import pickle #from Python standard library (allowed)

class Node:
#Tree node

    def __init__(self):
        self.true_child = self.false_child = None   #left and right children initially pointing to None.


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
        final_depth = None  #used in case final tree depth is < max depth
    
    #save model to text file after training
    def save_tree (self):
        pass

    #rebuild tree from saved model for predicting
    def load_tree (self, filepath):
        pass

    #recursively add nodes to list of lists
    @classmethod
    def tree_to_list(cls, node, tree_list, max_level, level, num):
        if node is None or level > max_level:   #cease recursion at end of tree or sufficient depth
            return tree_list
        else:
            tree_list[level-1][num] = f"Emitter {node.attribute} < {node.value}" if type(node) is Decision else f"Room {node.room}" #add label
            tree_list = DecisionTree.tree_to_list(node.true_child, tree_list, max_level, level+1, 2*num) #level-order node number = 2*num for left branches
            tree_list = DecisionTree.tree_to_list(node.false_child, tree_list, max_level, level+1, 2*num + 1) #level-order node number = 2*num + 1 for right branches
        return tree_list
    
    #traverses tree for prediction
    def search_tree(self, node, test_vals):
        if type(node) is Leaf:
            return node.room    #return final room prediction from leaf of tree
        elif test_vals[node.attribute] < node.value:    #check if (test) wifi strength on emitter (from tree) is less than node's decision value 
            return self.search_tree(node.true_child, test_vals) #go down left recursively
        else:
            return self.search_tree(node.true_child, test_vals) #go down right recursively
    
    def print_tree (self, depth=999999):
        list_depth = min(depth, self.final_depth) #print depth can be overwritten by arg
        tree_list = [[None for node in range(2**level)] for level in range(list_depth)] #blank nested list with perfect tree size
        tree_list = DecisionTree.tree_to_list(self.root, tree_list, list_depth, level=1, num=0)  #traverse tree and add to list
        print("Tree list:\n")
        for row in tree_list:
            print(row)
        plot_tree(tree_list)    #separate plot script


class Classifier:
#contains functions and class variables outside the DecisionTree scope

    max_depth = None    #early stopping criteria for trees
    dataset = None  #full data array from file

    # @classmethod
    # def entropy(cls, x):
    #     #x is a np.ndarray of size (N, K)
    #     values, count = np.unique(x, return_counts=True)
    #     entropy = 0

    #     for i in count:
    #         entropy += (-i / np.sum(count)) * np.log2(i / np.sum(count))

    #     return entropy

    @classmethod 
    def entropy_calc(cls, split_array):       # calculation of the entropy for a specific column split (top or bottom)
        unique, counts = np.unique(split_array[:,1], return_counts=True) #frequency of each room in array
        sum = np.sum(counts)    #total number of rooms (equals length of split_array along axis 0)
        proportions = counts / sum    #proportions of each room in the set
        entropies = -1 * proportions * np.log2(proportions) 
        entropy = np.sum(entropies)
        return entropy
    
    @classmethod
    def find_split(cls, dataset):   # entropy_for_all_columns
        lowest_entropy = 999999

        numcol = np.size(dataset,1)    #number of columns 
        for i in range(numcol-1):
            # print("i", i) #######
            min_entropy = 999999
            min_split = 0
            data = dataset[:,[i,-1]]    # extract the router column and label
            sorted_data = data[data[:, 0].argsort()]    # sort according to the router column values
            # print("sorted", sorted_data.shape)#####
            # print("unsorted", data)######
            # print("sorted", sorted_data)#####
            

            for a in range(len(sorted_data)-1):
                fsplit = sorted_data[:a+1,:]    # first/top split
                ssplit = sorted_data[a+1:,:]    # second/bottom split
                # print("a", a)
                sum_entropy = Classifier.entropy_calc(fsplit) + Classifier.entropy_calc(ssplit) # sum entropies which is used in information gain
                
                # print("entropy", sum_entropy)
                if sum_entropy < min_entropy:   # Checks if the entropy that was just calculated is lower than the lowest so far
                    min_entropy = sum_entropy   # Replaces the value of the return variable with the entropy that was just calculated
                    min_split = fsplit[a]       # Shows which split value gave the lowest entropy sum
            # print(str(min_entropy) + " smth " + str(min_split))

        
            if min_entropy < lowest_entropy:   # Checks if the entropy that was just calculated is lower than the lowest so far
                lowest_entropy = min_entropy   # Replaces the value of the return variable with the entropy that was just calculated
                lowest_split = min_split[0]       # Shows which split value gave the lowest entropy sum
                min_router = i                 # Stores the index of the router with the best split so far

            # print(str(min_router) + " smth " + str(lowest_split))
        return min_router, lowest_split   # Returns the min router(column index) and the split(row index) for this.
                
                
                
                



    # @classmethod
    # def entropy_full_column(dataset):
    #     # each col represents a different attribute
    #     col = np.size(dataset,1) ####### ==6
    #     entropies = []
    #     for i in range(col-1):  #subtract 1 to ignore the last column which is the room label
    #         column = dataset[:,i]
    #         entropy = 0
    #         unique, count = np.unique(column, return_counts=True)   #check which is unique and count the number of each unique val
    #         size = np.sum(count)  #total number of outcomes
    #         for j in count:
    #             prob = j / size 
    #             entropy += -1 * prob * np.log2(prob)
    #         entropies.append(entropy)   # gives a list of entropies for each column
        
        
    #     return entropies

    # def entropy_split_column(column_data):
    #     labels = 

    
    # @classmethod
    # def find_split(cls, data):
    #     # attribute = np.random.randint(data.shape[1]-1) #random column ##########
    #     # value = np.random.choice(data[:,attribute]) #random number from column #########
        
    #     entropies = Classifier.find_split(data) #stores the split decision attribute (the specific router in question)
    #     np_entropies = np.array(entropies)
    #     attribute = np.argmin(np_entropies)     # gives the index (column number) of the min entropy
    #     value = np.nanmin(np_entropies)     # gives the min value in the np array
    #     return attribute, value

    #recursive function which constructs tree and returns subtree root node
    @classmethod
    def decision_tree_learning (cls, data, depth):
        if len(data) == 0: ######temp for random split
            return Leaf(0), depth ######temp for random split

        room_labels, label_counts = np.unique(data[:, 7], return_counts=True) #get room labels and frequencies present in current subset
        if len(room_labels) == 1 or depth == Classifier.max_depth:    #if all samples from the same room or max_depth reached (early stopping)
            room_plurality = room_labels[label_counts==max(label_counts)][0]   #predicted room is mode of room labels
            leaf_node = Leaf(room_plurality)    #create leaf node with this room prediction
            return leaf_node, depth     #return leaf node and current depth to parent node
        else:
            attribute, value = Classifier.find_split(data)    #find optimal attribute and value to split by for this subset
            decision_node = Decision(attribute, value)        #create new node based on split choices
            true_subset = data[data[:, attribute]<value]      #subset which follows the condition "attribute < value"
            false_subset = data[data[:, attribute]>=value]    #complement set (doesn't follow condition)
            decision_node.true_child, true_subtree_depth = Classifier.decision_tree_learning(true_subset, depth+1) #recursive call on true side of dataset
            decision_node.false_child, false_subtree_depth = Classifier.decision_tree_learning(false_subset, depth+1) #recursive call on false side of dataset
            return (decision_node, max(true_subtree_depth, false_subtree_depth)) #returns node and current max depth to parent node

    #callable by other functions to commence training
    @classmethod
    def fit (cls, dataset_filepath, max_depth):
        Classifier.max_depth = max_depth    #tree will stop constructing when this depth is reached
        Classifier.dataset = np.loadtxt(dataset_filepath).astype(np.int64)    #load data from text file into integer numpy array
        tree = DecisionTree()     #instantiate blank tree
        tree.root, tree.final_depth = Classifier.decision_tree_learning(Classifier.dataset, depth=1)  #start recursive training process
        return tree

    #querying algorithm to traverse through the decision tree to allocate room numbers to test data entered
    @classmethod
    def predict (cls, tree, test_set):
        predictions = []
        test_set = np.array([test_set]) if test_set.ndim == 1 else test_set #ensure that test_set has n tests in it (2D array)
        for test in test_set:   #complete prediction for each test
            predictions.append(tree.search_tree(tree.root, test))
        predictions = np.array(predictions)
        return predictions  #1D array of class labels (rooms) for each test

#default main when file ran individually
if __name__ == "__main__":
    tree = Classifier.fit(r'intro2ML-coursework1\wifi_db\noisy_dataset.txt', max_depth=10)
    print("Prediction: ", Classifier.predict(tree, np.array([-64, -56, -61, -66, -71, -82, -81])))
    tree.print_tree()

    #-64 -56 -61 -66 -71 -82 -81 1
