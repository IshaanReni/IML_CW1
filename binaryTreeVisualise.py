import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches

# Reference: https://stackoverflow.com/questions/59028711/plotting-a-binary-tree-in-matplotlib
# https://stackoverflow.com/questions/14531346/how-to-add-a-text-into-a-rectangle

t_levels = 5                                    # all sizes relative to the nr of levels
width_dist = t_levels**2                        # distance between parent node and child node (x axis)
depth_dist = 20                                 # height between levels
label_Width = t_levels*2.2                      # width of the rectangle label for each node
label_Height = t_levels*0.45                    # height of the rectangle label for each node
font_size = t_levels*1.2                        #font size on the label
nodes = []

def binaryTree_gen(levels, x, y, width):        # assume complete tree given
    # spaceBetween = (t_levels-levels+1)*width
    segments = []
    xl = x - width/2                            # calculating the coords for the left and right child node
    yl = y - depth_dist
    xr = x + width/2
    yr = y - depth_dist
    
    if t_levels == levels:                      # base case to create label node for the initial node
        segments.append([[x,y], ])
        nodes.append(patches.Rectangle((x - label_Width/2,y - label_Height/2), label_Width, label_Height))

    segments.append([[x, y], [xl, yl]])         # adding the edges/sergments for connecting parent to the two children
    segments.append([[x, y], [xr, yr]])

    xl_rectangle = xl - label_Width/2           # calculating label coords for every child node and adding them into all nodes array
    yl_rectangle = yl - label_Height/2
    nodes.append(patches.Rectangle((xl_rectangle, yl_rectangle), label_Width, label_Height, label="left"))
    nodes.append(patches.Rectangle((xr - label_Width/2, yr - label_Height/2), label_Width, label_Height, label="right"))
    if levels > 1:                              # recursive call if were not on the final level
        segments += binaryTree_gen(levels - 1, xl, yl, width/2)
        segments += binaryTree_gen(levels - 1, xr, yr, width/2)

    return segments


def plot_tree (tree_list):

    segs = binaryTree_gen(t_levels, 0, 0, width_dist) # initial call of the gen tree function

    fig, ax = plt.subplots()                        #set axis contraints
    ax.set_ylim(-(t_levels * depth_dist + 5), 5)
    ax.set_xlim(-2*width_dist, 2*width_dist)

    flat_tree_list = [node for level in tree_list for node in level]

    for i, r in enumerate(nodes):
        rx, ry = r.get_xy()
        cx = rx + r.get_width()/2.0
        cy = ry + r.get_height()/2.0
        label = flat_tree_list[i]
        # if label == None:
        #     del segs[i]
        #     del segs[i+1]
        #     del nodes[i]
        #     del nodes[i+1]
        ax.annotate(label, (cx, cy), color='w', weight='bold', 
                    fontsize=font_size, ha='center', va='center')   # adding collection elements for text labels

    # creating matplotlib collection elements for both edges and nodes
    line_segments = LineCollection(segs, linewidths=1, colors='red', linestyle='solid')
    tree_nodes = PatchCollection(nodes, linewidth=1, edgecolor='green', facecolor='green')

    ax.add_collection(line_segments)                # adding collection elements for edges
    ax.add_collection(tree_nodes)                   # adding collection elements for nodes

    plt.show()