import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches

# Reference: https://stackoverflow.com/questions/59028711/plotting-a-binary-tree-in-matplotlib
# https://stackoverflow.com/questions/14531346/how-to-add-a-text-into-a-rectangle

nodes = []


# assume complete tree given
def binaryTree_gen(levels, x, y, width, t_levels, depth_dist, label_Width, label_Height):
    # spaceBetween = (t_levels-levels+1)*width
    segments = []
    # calculating the coords for the left and right child node
    xl = x - width/2
    yl = y - depth_dist
    xr = x + width/2
    yr = y - depth_dist

    if t_levels == levels:                      # base case to create label node for the initial node
        segments.append([[x, y], ])
        nodes.append(patches.Rectangle(
            (x - label_Width/2, y - label_Height/2), label_Width, label_Height))

    # adding the edges/sergments for connecting parent to the two children
    segments.append([[x, y], [xl, yl]])
    segments.append([[x, y], [xr, yr]])

    # calculating label coords for every child node and adding them into all nodes array
    xl_rectangle = xl - label_Width/2
    yl_rectangle = yl - label_Height/2
    nodes.append(patches.Rectangle((xl_rectangle, yl_rectangle),
                 label_Width, label_Height, label="left"))
    nodes.append(patches.Rectangle((xr - label_Width/2, yr -
                 label_Height/2), label_Width, label_Height, label="right"))
    if levels > 1:                              # recursive call if were not on the final level
        segments += binaryTree_gen(levels - 1, xl, yl, width/2,
                                   t_levels, depth_dist, label_Width, label_Height)
        segments += binaryTree_gen(levels - 1, xr, yr, width/2,
                                   t_levels, depth_dist, label_Width, label_Height)

    return segments


def plot_tree(inorder_list, t_levels):
    # distance between parent node and child node (x axis)
    width_dist = t_levels**2
    depth_dist = 20                                 # height between levels
    # width of the rectangle label for each node
    label_Width = t_levels*2.2
    # height of the rectangle label for each node
    label_Height = t_levels*0.45
    font_size = t_levels*1.2  # font size on the label

    segs = binaryTree_gen(t_levels, 0, 0, width_dist, t_levels, depth_dist,
                          label_Width, label_Height)  # initial call of the gen tree function

    fig, ax = plt.subplots()  # set axis contraints
    ax.set_ylim(-(t_levels * depth_dist + 5), 5)
    ax.set_xlim(-2*width_dist, 2*width_dist)

    for i, r in enumerate(nodes):
        rx, ry = r.get_xy()
        cx = rx + r.get_width()/2.0
        cy = ry + r.get_height()/2.0
        print("Label:", inorder_list[i], "with index: ", i)
        label = inorder_list[i]
        # if label == None:
        #     del segs[i]
        #     del segs[i+1]
        #     del nodes[i]
        #     del nodes[i+1]
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
