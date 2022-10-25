import numpy as np  #allowed
from preorderTreeVisualise import plot_preorder_tree #own file
import sys #from Python standard library (allowed)
# from convertToPreorder import convert_to_preorder_array #own file

np.set_printoptions(threshold=sys.maxsize) #print whole arrays


class Node:
#Tree node

    def __init__(self, data):
        self.true_child = self.false_child = None   #left and right children initially pointing to None.
        self.data = data    #subset at this point is used later for pruning (mode)

class Decision (Node):
#for non-leaves. Inherits Node class. Stores attribute and value of decision

    def __init__(self, data, attribute, value):
        super().__init__(data)          #run inherited __init__() first
        self.attribute = attribute  #which wifi emitter to split by (column index)
        self.value = value          #value by which to split into two subsets, for a given attribute

    #backup branches means we don't have to work with a deepcopy of the original tree
    #this self node will be 2 levels above the current leaves
    def prune_child(self, child):
        if child == True:
            backup_branch = self.true_child #transfer the branch to be pruned into a backup pointer (in case accuracy decreases)
            data = self.true_child.data #use child's data to choose class of leaf
            room_labels, label_counts = np.unique(data[:, -1], return_counts=True) #get room labels and frequencies present in current subset
            room_plurality = room_labels[label_counts==max(label_counts)][0]   #predicted room is mode of room labels
            self.true_child = Leaf(data, room_plurality)    #replace child with leaf having this room prediction
        elif child == False:
            backup_branch = self.false_child #transfer the branch to be pruned into a backup pointer (in case accuracy decreases)
            data = self.false_child.data #use child's data to choose class of leaf
            room_labels, label_counts = np.unique(data[:, -1], return_counts=True) #get room labels and frequencies present in current subset
            room_plurality = room_labels[label_counts==max(label_counts)][0]   #predicted room is mode of room labels
            self.false_child = Leaf(data, room_plurality)    #replace child with leaf having this room prediction
        return backup_branch    #used for undo
    
    #used when accuracy decreased after pruning
    def undo_prune (self, child, backup_branch):
        if child == True:
            self.true_child = backup_branch
        elif child == False:
            self.false_child = backup_branch


class Leaf (Node):
#for leaves. Inherits Node class. Stores class label (room no.) for prediction
    
    def __init__(self, data, room):
        super().__init__(data)          #run inherited __init__() first
        self.room = room            #final predicted class label


class DecisionTree:
#Tree stores root node

    #constructor for a tree
    def __init__(self):
        self.root = None            #beginning of tree
        self.prev_column = 99999    #to ensure we split by a different column each time in the tree

    #more efficient method of counting tree stats than making tree_lists every time.
    def tree_properties(self, node, nodes=1, leaves=0, depth=1):
        if type(node) is Leaf:
            return nodes, leaves+1, depth
        nodes, leaves, true_subtree_depth = self.tree_properties(node.true_child, nodes+1, leaves, depth+1)
        nodes, leaves, false_subtree_depth = self.tree_properties(node.false_child, nodes+1, leaves, depth+1)
        return nodes, leaves, max(true_subtree_depth, false_subtree_depth)

    #call to get number of nodes of tree as it stands
    def get_nodes(self):
        return self.tree_properties(self.root)[0]
    
    #call to get number of leaves of tree as it stands
    def get_leaves(self):
        return self.tree_properties(self.root)[1]

    #call to get depth of tree as it stands
    def get_depth(self):
        return self.tree_properties(self.root)[2]

    #traverses tree for prediction
    def search_tree(self, node, test_vals):
        if type(node) is Leaf:
            return node.room                                    #return final room prediction from leaf of tree
        elif test_vals[node.attribute] > node.value:            #check if (test) wifi strength on emitter (from tree) is greater than node's decision value 
            return self.search_tree(node.true_child, test_vals) #go down left recursively
        else:
            return self.search_tree(node.false_child, test_vals) #go down right recursively

    #recursively add nodes to list of lists and preorder list
    @classmethod
    def tree_lists(cls, node, tree_list, preorder_list, max_level, level, num):
        if node is None or level > max_level:   #cease recursion at end of tree or sufficient depth
            diff = max_level - (level-1)        # Need to insert Nones for perfect tree. dictates the number of compeletely None levels we need to insert
            for i in range(diff):               # for each remaining level
                for j in range(2**i): 
                    preorder_list.append(None)  # Place down the appropriate amount of Nones depending on the level we're on right now
            return tree_list, preorder_list     #returns a human-readbale list and a plottable list (respectively) 
        else:
            label_string = f"Emitter {node.attribute} > {node.value}" if type(node) is Decision else f"Room {node.room}"
            tree_list[level-1][num] = label_string      # list of decision labels/rooms for human readability
            preorder_list.append(label_string)
            tree_list, preorder_list = DecisionTree.tree_lists(node.true_child, tree_list, preorder_list, max_level, level+1, 2*num) #level-order node number = 2*num for left branches
            tree_list, preorder_list = DecisionTree.tree_lists(node.false_child, tree_list, preorder_list, max_level, level+1, 2*num + 1) #level-order node number = 2*num + 1 for right branches
        return tree_list, preorder_list
    
    #creates a nested level-order tree list for human-readability, and an inorder list for plotting
    def print_tree (self, depth=999999):
        list_depth = min(depth, self.get_depth())     #print depth can be overwritten by arg
        tree_list = [[None for node in range(2**level)] for level in range(list_depth)] #blank nested list with perfect tree size
        tree_list, preorder_list = DecisionTree.tree_lists(self.root, tree_list, [], list_depth, level=1, num=0)  #traverse tree and add to list
        #print("Tree list:\n")
        #for row in tree_list:   #human readable, level-order
            #print(row)
        #print("Preorder list\n")
        #print(preorder_list)     #to send for plotting
        plot_preorder_tree(preorder_list, list_depth)
        # converted_preorder_list = convert_to_preorder_array(tree_list)        #converted 2D array into a preorder 1D array for the plot
        # plot_preorder_tree(converted_preorder_list, list_depth)               #separate plot script
        # plot_preorder_tree([1,2,None,4,5,6,None], 3) #separate preorder plot script


class Classifier:
#contains functions and class variables outside the DecisionTree scope

    max_depth = None    #early stopping criteria for trees
    dataset = None      #full data array from file

    @classmethod 
    def entropy_calc(cls, split_array):       # calculation of the entropy for a specific column split (top or bottom)
        unique, counts = np.unique(split_array[:,1], return_counts=True)    #frequency of each room in array
        sum = np.sum(counts)                                #total number of rooms (equals length of split_array along axis 0)
        proportions = counts / sum                          #proportions of each room in the set
        entropies = -1 * proportions * np.log2(proportions) #entropies for each room in the set
        entropy = np.sum(entropies)                         #total entropy for this split
        return entropy
    
    @classmethod
    def find_split(cls, dataset, tree):   # entropy_for_all_columns
        min_entropy = 999999
        min_router = -1 #default value which should be overwritten
        min_split = -1  #default value which should be overwritten

        numcols = [i for i in range(np.size(dataset,1)-1)]     #numbers 0..6
        try: numcols.pop(tree.prev_column)       #don't check previous column
        except: pass
        for emitter in numcols:                #for each column (not including previous)
            min_col_entropy = 999999                    #will be replaced with lowest entropy
            data = dataset[:,[emitter,-1]]              # extract the router column and label
            sorted_data = data[data[:, 0].argsort()[::-1]]   # sort according to the router column values in descending order(makes splits more efficient)         
            # #print("Emitter:: ", emitter)
            for split_point_index in range(1,len(sorted_data)):         #for each emitter value
                greater_split_array = sorted_data[:split_point_index]      # top half of the split column in array form
                less_split_array = sorted_data[split_point_index:]      # bottom half of the split column in array form
                # #print("l_split_arr:", less_split_array)
                # #print("g_split_arr:", greater_split_array)
                sum_entropy = (len(greater_split_array)/(len(sorted_data)))*Classifier.entropy_calc(greater_split_array) + (len(less_split_array)/(len(sorted_data)))*Classifier.entropy_calc(less_split_array) # weighted sum entropies which is used in information gain
                
                # #print("inside:", str(less_split_array))
                # #print(sum_entropy < min_col_entropy,"=>", sum_entropy,",", min_col_entropy)
                if sum_entropy <= min_col_entropy:               # overwrites previous entropy if less
                    min_col_entropy = sum_entropy               # Minimum entropy for this emitter
                    min_col_split = less_split_array[0][0]     # Which split value gave the lowest entropy sum for this emitter
                # #print("min entropy: "+ str(min_entropy) + "; min_split (row value where to split): " + str(min_split)) ######

                # #print(min_col_entropy < min_entropy,"=>", min_col_entropy,",", min_entropy)
                # #print(history, [min_router, min_split])
                # #print([min_router, min_split] in history)
            if min_col_entropy < min_entropy:   # Checks if the entropy that was just calculated is lower than the lowest so far
                min_entropy = min_col_entropy   # Replaces the value of the return variable with the entropy that was just calculated
                min_split = min_col_split       # Which split value gave the lowest entropy sum overall
                min_router = emitter            # Stores the index of the router with the best split so far

            # #print("min router: "+ str(min_router) + "; min_split (row value where to split): " +str(min_split)) ######
        # #print("outside less split:", str(sorted_data[:min_split]))
        # #print("outside sorted data:", str(sorted_data))
        # #print("outside dataset:", str(dataset[:,-1]))
        tree.prev_column = min_router #set previous column to the emitter we are splitting by in this iteration
        assert(min_router!=-1 and min_split!=-1), f"Decision tree training error: decision={min_router, min_split}"#+ ("l_split_arr:", str(less_split_array))+ ("g_split_arr:", str(greater_split_array)) #error handling
        ##print("g_split_arr:", greater_split_array)
        return min_router, min_split   # Returns the min router(column index) and the split(row index) for this.
        
    #recursive function which constructs tree and returns subtree root node
    @classmethod
    def decision_tree_learning (cls, data, tree, depth=1):   #depth 1 by default
        room_labels, label_counts = np.unique(data[:, -1], return_counts=True) #get room labels and frequencies present in current subset
        if len(room_labels) == 1 or depth == Classifier.max_depth:    #if all samples from the same room or max_depth reached (early stopping)
            room_plurality = room_labels[label_counts==max(label_counts)][0]   #predicted room is mode of room labels
            leaf_node = Leaf(data, room_plurality)    #create leaf node with this room prediction
            return leaf_node, depth     #return leaf node and current depth to parent node
        else:
            attribute, value = Classifier.find_split(data, tree)    #find optimal attribute and value to split by for this subset
            decision_node = Decision(data, attribute, value)        #create new node based on split choices
            #print("Attribute:", attribute)
            #print("Value:", value)
            true_subset = data[data[:, attribute]>value]      #subset which follows the condition "attribute > value"
            #print("true_subset:", true_subset)
            false_subset = data[data[:, attribute]<=value]    #complement set (doesn't follow condition)
            #print("false_subset:", false_subset)
            #print("-------------------")
            # print("t:", true_subset[:,-1])###
            # print("f:", false_subset[:,-1])###
            if (not true_subset.tolist()) or (not false_subset.tolist()):     #entropy is so similar that one partition is empty
                room_plurality = room_labels[label_counts==max(label_counts)][0]   #predicted room is mode of room labels
                leaf_node = Leaf(data, room_plurality)    #create leaf node with this room prediction
                return leaf_node, depth     #return leaf node and current depth to parent node
            else:   #recurse otherwise
                decision_node.true_child, true_subtree_depth = Classifier.decision_tree_learning(true_subset, tree, depth+1) #recursive call on true side of dataset
                decision_node.false_child, false_subtree_depth = Classifier.decision_tree_learning(false_subset, tree, depth+1) #recursive call on false side of dataset
                return decision_node, max(true_subtree_depth, false_subtree_depth) #returns node and current max depth to parent node

    #recursive function which prunes a decision tree while calculating accuracy at each step
    @classmethod
    def decision_tree_pruning (cls, tree, node):
        #######incomplete test of prune_child(). This should prune everything
        if type(node) is Leaf:
            return False
        prune_signal = Classifier.decision_tree_pruning(tree, node.true_child)
        if prune_signal == True:
            backup_branch = node.prune_child(child=True)
        prune_signal = Classifier.decision_tree_pruning(tree, node.false_child)
        if prune_signal == True:
            backup_branch = node.prune_child(child=False)
        if type(node.true_child) is Leaf and type(node.false_child) is Leaf:
            return True

    #callable by other functions to commence training
    @classmethod
    def fit (cls, dataset, max_depth=99):
        Classifier.max_depth = max_depth    #tree will stop constructing when this depth is reached
        Classifier.dataset = dataset
        tree = DecisionTree()     #instantiate blank tree
        tree.root = Classifier.decision_tree_learning(Classifier.dataset, tree)[0]  #start recursive training process
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
    dataset = np.loadtxt(r'intro2ML-coursework1/wifi_db/noisy_dataset.txt').astype(np.int64)    #load data from text file into integer numpy array
    tree = Classifier.fit(dataset)

    #print("Prediction: ", Classifier.predict(tree, np.array([-64, -56, -61, -66, -71, -82, -81])))
    tree.print_tree()

    #-64 -56 -61 -66 -71 -82 -81 1


