import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
# from matplotlib import colors as mcolors

# Reference: https://stackoverflow.com/questions/59028711/plotting-a-binary-tree-in-matplotlib
# https://stackoverflow.com/questions/14531346/how-to-add-a-text-into-a-rectangle

width_dist = 10
depth_dist = 10
levels = 4
label_Width = 4
label_Height = 1
nodes = []

def binaryTree_gen(levels, x, y, width):
    segments = []
    xl = x - width / 2
    yl = y - depth_dist
    xr = x + width / 2 
    yr = y - depth_dist
    segments.append([[x, y], [xl, yl]])
    segments.append([[x, y], [xr, yr]])
    nodes.append(patches.Rectangle((xl - label_Width/2, yl - label_Height/2), label_Width, label_Height, label="left"))
    nodes.append(patches.Rectangle((xr - label_Width/2, yr - label_Height/2), label_Width, label_Height, label="right"))
    if levels > 1:
        segments += binaryTree_gen(levels - 1, xl, yl, width / 2)
        segments += binaryTree_gen(levels - 1, xr, yr, width  / 2)

    return segments


def plot_tree (tree_list):

    segs = binaryTree_gen(levels, 0, 0, width_dist)

    # colors = [mcolors.to_rgba(c)
    #           for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    line_segments = LineCollection(segs, linewidths=1, colors='red', linestyle='solid')
    tree_nodes = PatchCollection(nodes, linewidth=1, edgecolor='green', facecolor='none')


    fig, ax = plt.subplots()
    ax.set_ylim(-(levels * depth_dist + 1), 1)
    ax.set_xlim(-1.5*width_dist, 1.5*width_dist)
    ax.add_collection(line_segments)
    ax.add_collection(tree_nodes)

    flat_tree_list = [node for level in tree_list for node in level]

    for i, r in enumerate(nodes):
        ax.add_artist(r)
        rx, ry = r.get_xy()
        cx = rx + r.get_width()/2.0
        cy = ry + r.get_height()/2.0
        label = flat_tree_list[i]
        ax.annotate(label, (cx, cy), color='w', weight='bold', 
                    fontsize=6, ha='center', va='center')

    plt.show()