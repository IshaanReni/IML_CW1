# Python program to for tree traversals
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches

# A class that represents an individual node in a
nodes = []  # Rectangle objects with (x,y)

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# A function to do preorder tree traversal


def preorder_binaryTree_gen(inorder_list, levels, x, y, width, depth_dist, label_Width, label_Height):
        # calculating the coords for the left and right child node
        xl = x - width/2
        yl = y - depth_dist
        xr = x + width/2
        yr = y - depth_dist

        label = str(width)+","+str(len(nodes))
        print('ADD node_index: ',len(nodes), " with label ", inorder_list[len(nodes)])
        if inorder_list[len(nodes)] != None:
            print('--------------------node not None: GOOD')
            nodes.append(patches.Rectangle(
                    (x - label_Width/2, y - label_Height/2), label_Width, label_Height, label=label)) # label is saved as a string
        else:
            nodes.append(None)
            print('--------------------node is  None: None')

        if levels > 1:
            # Then recur on left child
            preorder_binaryTree_gen(inorder_list, levels - 1, xl, yl, width/2,
                                 depth_dist, label_Width, label_Height)

            # Finally recur on right child
            preorder_binaryTree_gen(inorder_list, levels - 1, xr, yr, width/2,
                                 depth_dist, label_Width, label_Height)


def plot_edges(nodes, inorder_list, depth_dist, label_Width, label_Height):
    segments = []

    for n in nodes:
        if n != None:
            x_n, y_n = n.get_xy()
            #possible children
            max_children_nodes = 2
            # print(n.get_label().split(',')[0])
            x_child_l = x_n - float(n.get_label().split(',')[0])/2 
            y_child_l = y_n - depth_dist
            x_child_r = x_n + float(n.get_label().split(',')[0])/2 
            y_child_r = y_n - depth_dist
            # print("node_index:",n.get_label().split(',')[1],
            #         ",nodes checking for edges: ",x_n,";",y_n, " - children: left::", x_child_l,";",y_child_l,
            #         " right::", x_child_r,";",y_child_r )
            for n2 in nodes:
                if max_children_nodes > 0 and n2 != None:
                    x_n2, y_n2 = n2.get_xy()
                    # print("label of n2: ",inorder_list[int(n2.get_label().split(',')[1])])
                    if inorder_list[int(n2.get_label().split(',')[1])] != None:
                        if x_n2 == x_child_l and y_n2 == y_child_l:
                            segments.append([[x_n + label_Width/2, y_n], [x_n2 + label_Width/2, y_n2 + label_Height/2]])
                            max_children_nodes=max_children_nodes-1
                        elif x_n2 == x_child_r and y_n2 == y_child_r:
                            segments.append([[x_n + label_Width/2, y_n], [x_n2 + label_Width/2, y_n2 + label_Height/2]])
                            max_children_nodes=max_children_nodes-1
                else:
                    break

    return segments


def plot_preorder_tree(inorder_list, t_levels):
    # distance between parent node and child node (x axis)
    width_dist = t_levels**2
    depth_dist = 20                                 # height between levels
    # width of the rectangle label for each node
    label_Width = t_levels*2.2
    # height of the rectangle label for each node
    label_Height = t_levels*0.45
    font_size = t_levels*1.2  # font size on the label

    print(inorder_list)

    preorder_binaryTree_gen(inorder_list, t_levels, 0, 0, width_dist, depth_dist, label_Width, label_Height)  # initial call of the gen tree function

    fig, ax = plt.subplots()  # set axis contraints
    ax.set_ylim(-(t_levels * depth_dist + 5), 5)
    ax.set_xlim(-2*width_dist, 2*width_dist)

    non_none_nodes = nodes
    none_n_indexes = []

    for i, r in enumerate(nodes):
        if r != None:
            rx, ry = r.get_xy()
            cx = rx + r.get_width()/2.0
            cy = ry + r.get_height()/2.0
            print("Label:", inorder_list[i], "with index: ", i)
            label = inorder_list[i]
            ax.annotate(label, (cx, cy), color='w', weight='bold',
                        fontsize=font_size, ha='center', va='center')   # adding collection elements for text labels
        else:
            none_n_indexes.append(i)                # store the indexes of which nodes are None, to remove later
    
    reverse_none_n_indexes = none_n_indexes[::-1]   # inverting the none indexes array to start from the back of the nodes array when removing the None's
    for r_n_index in reverse_none_n_indexes:
        del non_none_nodes[r_n_index]               # removing the nodes with value None from the copy array of the nodes

    segs = plot_edges(non_none_nodes, inorder_list, depth_dist, label_Width, label_Height) # the function has checked to ignore nodes that are None

    # creating matplotlib collection elements for both edges and nodes
    line_segments = LineCollection(
        segs, linewidths=1, colors='red', linestyle='solid')
    tree_nodes = PatchCollection(
        non_none_nodes, linewidth=1, edgecolor='green', facecolor='green') # ERROR if the tree containing unreachable branches due to None

    # adding collection elements for edges
    ax.add_collection(line_segments)
    # adding collection elements for nodes
    ax.add_collection(tree_nodes)

    plt.show()
