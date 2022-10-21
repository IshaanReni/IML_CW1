# Python program to for tree traversals
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches

# A class that represents an individual node in a
nodes = []  # [label, x, y]
# Binary Tree


class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# A function to do preorder tree traversal


def preorder_binaryTree_gen(inorder_list, levels, x, y, width, t_levels, depth_dist, label_Width, label_Height):
        # calculating the coords for the left and right child node
        xl = x - width/2
        yl = y - depth_dist
        xr = x + width/2
        yr = y - depth_dist

        nodes.append(patches.Rectangle(
                (x - label_Width/2, y - label_Height/2), label_Width, label_Height))

        if levels > 1:
            # Then recur on left child
            preorder_binaryTree_gen(inorder_list, levels - 1, xl, yl, width/2,
                                t_levels, depth_dist, label_Width, label_Height)

            # Finally recur on right child
            preorder_binaryTree_gen(inorder_list, levels - 1, xr, yr, width/2,
                                t_levels, depth_dist, label_Width, label_Height)


def plot_edges(levels, x, y, width, t_levels, depth_dist):
    segments = []
    # calculating the coords for the left and right child node
    xl = x - width/2
    yl = y - depth_dist
    xr = x + width/2
    yr = y - depth_dist

    if levels > 1:
        # adding the edges/sergments for connecting parent to the two children
        segments.append([[x, y], [xl, yl]])
        # Then recur on left child
        segments += plot_edges(levels - 1, xl, yl, width/2, t_levels, depth_dist)

        segments.append([[x, y], [xr, yr]])
        # Finally recur on right child
        segments += plot_edges(levels - 1, xr, yr, width/2, t_levels, depth_dist)

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

    preorder_binaryTree_gen(inorder_list, t_levels, 0, 0, width_dist, t_levels, depth_dist, label_Width, label_Height)  # initial call of the gen tree function
    segs = plot_edges(t_levels, 0, 0, width_dist, t_levels, depth_dist)

    fig, ax = plt.subplots()  # set axis contraints
    ax.set_ylim(-(t_levels * depth_dist + 5), 5)
    ax.set_xlim(-2*width_dist, 2*width_dist)

    for i, r in enumerate(nodes):
        rx, ry = r.get_xy()
        cx = rx + r.get_width()/2.0
        cy = ry + r.get_height()/2.0
        print("Label:", inorder_list[i], "with index: ", i)
        label = inorder_list[i]
        if label == None:
            del nodes[i]
            # to add, delete edge that ends in the coords of the node with value None
        ax.annotate(label, (cx, cy), color='w', weight='bold',
                    fontsize=font_size, ha='center', va='center')   # adding collection elements for text labels

    # creating matplotlib collection elements for both edges and nodes
    line_segments = LineCollection(
        segs, linewidths=1, colors='red', linestyle='solid')
    tree_nodes = PatchCollection(
        nodes, linewidth=1, edgecolor='green', facecolor='green')

    # adding collection elements for edges
    ax.add_collection(line_segments)
    # adding collection elements for nodes
    ax.add_collection(tree_nodes)

    plt.show()
