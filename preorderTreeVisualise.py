# Python program to for tree traversals
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import math

# # A function to do preorder tree traversal

def preorder_binaryTree_gen(order_list, nodes, levels, x, y, width, depth_dist, label_Width, label_Height):
        # calculating the coords for the left and right child node
        xl = x - width/2
        yl = y - depth_dist
        xr = x + width/2
        yr = y - depth_dist

        label = str(width)+","+str(len(nodes))
        #print('ADD node_index: ',len(nodes), " with label ", order_list[len(nodes)])
        if order_list[len(nodes)] != None:
            #print('--------------------node not None: GOOD')
            nodes.append(patches.Rectangle(
                    (x - label_Width/2, y - label_Height/2), label_Width, label_Height, label=label)) # label is saved as a string
        else:
            nodes.append(None)
            #print('--------------------node is  None: None')

        if levels > 1:
            # Then recur on left child
            preorder_binaryTree_gen(order_list, nodes, levels - 1, xl, yl, width/2,
                                 depth_dist, label_Width, label_Height)

            # Finally recur on right child
            preorder_binaryTree_gen(order_list, nodes, levels - 1, xr, yr, width/2,
                                 depth_dist, label_Width, label_Height)


def plot_edges(nodes, order_list, depth_dist, label_Width, label_Height):
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
            # print("node_index:",n.get_label().split(',')[1], ", label: ", order_list[int(n.get_label().split(',')[1])],
            #         ",nodes checking for edges: ",x_n,";",y_n, " - children: left::", x_child_l,";",y_child_l,
            #         " right::", x_child_r,";",y_child_r )
            for n2 in nodes:
                if max_children_nodes > 0 and n2 != None:
                    x_n2, y_n2 = n2.get_xy()
                    # print("label of n2: ",order_list[int(n2.get_label().split(',')[1])])
                    if order_list[int(n2.get_label().split(',')[1])] != None:
                        if round(x_n2,4) == round(x_child_l,4) and round(y_n2,4) == round(y_child_l,4):
                            segments.append([[x_n + label_Width/2, y_n], [x_n2 + label_Width/2, y_n2 + label_Height]])
                            max_children_nodes=max_children_nodes-1
                        elif round(x_n2,4) == round(x_child_r,4) and round(y_n2,4) == round(y_child_r,4):
                            segments.append([[x_n + label_Width/2, y_n], [x_n2 + label_Width/2, y_n2 + label_Height]])
                            max_children_nodes=max_children_nodes-1
                else:
                    break

    return segments


def plot_preorder_tree(order_list, t_levels):
    # A class that represents an individual node in a
    nodes = []  # Rectangle objects with (x,y)
    # distance between parent node and child node (x axis)
    width_dist = t_levels**(2)
    depth_dist = 20                                 # height between levels
    # width of the rectangle label for each node
    label_Width = t_levels*2.4 #25
    # height of the rectangle label for each node
    label_Height = t_levels*0.4 #7
    font_size = 6 #t_levels*1.2  # font size on the label

    # print(order_list)

    preorder_binaryTree_gen(order_list, nodes, t_levels, 0, 0, width_dist, depth_dist, label_Width, label_Height)  # initial call of the gen tree function

    fig, ax = plt.subplots()  # set axis contraints
    ax.set_ylim(-(t_levels * depth_dist + 5), 5)
    ax.set_xlim(-2*width_dist, 2*width_dist)

    not_none_nodes = nodes
    none_n_indexes = []

    for i, r in enumerate(nodes):
        if r != None:
            rx, ry = r.get_xy()
            cx = rx + r.get_width()/2.0
            cy = ry + r.get_height()/2.0
            # print("Label:", order_list[i], "with index: ", i)
            label = order_list[i]
            ax.annotate(label, (cx, cy), color='w', weight='bold',
                        fontsize=font_size, ha='center', va='center')   # adding collection elements for text labels
        else:
            none_n_indexes.append(i)                # store the indexes of which nodes are None, to remove later
    
    reverse_none_n_indexes = none_n_indexes[::-1]   # inverting the none indexes array to start from the back of the nodes array when removing the None's
    for r_n_index in reverse_none_n_indexes:
        del not_none_nodes[r_n_index]               # removing the nodes with value None from the copy array of the nodes

    segs = plot_edges(not_none_nodes, order_list, depth_dist, label_Width, label_Height) # the function has checked to ignore nodes that are None

    # creating matplotlib collection elements for both edges and nodes
    line_segments = LineCollection(
        segs, linewidths=1, colors='blue', linestyle='solid')
    tree_nodes = PatchCollection(
        not_none_nodes, linewidth=1, edgecolor='none', facecolor='green') # ERROR if the tree containing unreachable branches due to None

    # adding collection elements for edges
    ax.add_collection(line_segments)
    # adding collection elements for nodes
    ax.add_collection(tree_nodes)

    legend = (f"Tree depth: {t_levels}\n"
              f"Branches: {len(segs)}\n"
              f"Leaves: {math.ceil((len(segs)+2)/2)}")

    ax.text(-1.9*width_dist, -10, legend, style='italic',
        bbox={'facecolor': 'none', 'alpha': 0.5, 'pad': 10})

    plt.axis('off')
    plt.show()

    
