
#####################################################
# Extract the forest structure from RF trained in R #
#####################################################

import numpy as np

# Set indices to intuitive variables to improve readibility
l_node, r_node, split_f, status, pred = (0, 1, 2, 3, 4)

###################################################
# Node class which stores structure and node info #
###################################################
class Node:

    def __init__(self, n, f, l, s):
        self.num = n            # node number (index)
        self.feature = f        # feature name
        self.probs = {}         # prob of each class at that node
        self.count = {}         # count of each class at that node
        self.left_child = None  # left child node
        self.right_child = None # right child node
        self.path_len = l       # length of root-to-node subpath
        self.subpath = s        # list of nodes in root-to-leaf subpath

    # Set children nodes
    def set_children(self, l_child, r_child):
        self.left_child = l_child
        self.right_child = r_child

    # Update probability of class based on frequency of leaf nodes
    def update_probs(self):

        # Access class probs of children
        l = self.left_child.count
        r = self.right_child.count

        # Initialize classes as keys
        all_keys = set(l.keys()).union(set(r.keys()))

        # Compute freq of each class in leaf nodes based on children
        for k in all_keys:
            count = 0
            if k in l:
                count += l[k]
            if k in r:
                count += r[k]
            self.count[k] = count

        # Turn frequency count into probability
        total = float(sum(self.count.values()))
        self.probs = dict((k, float(self.count[k]) / total) for k in all_keys)

######################################################
# Tree class which stores paths, nodes and leaf info #
######################################################
class Tree:

    def __init__(self, block):
        self.d_block = block    # original data block from R describing tree
        self.nodes = []         # list of node objects
        self.paths = []         # list of features in all root-to-leaf paths
        self.preds = []         # list of leaf node class labels
        self.partial_path = []  # list of node objects being built out into a path

    # Parse data block to determine all paths
    def get_paths(self):

        # Start at root node
        node = 0

        # Access node number of left and right children from root
        left_node = int(self.d_block[(node, l_node)] - 1)
        right_node = int(self.d_block[(node, r_node)] - 1)

        # Access feature number at root
        root_f = int(self.d_block[(node, split_f)] - 1)

        # Initialize path to begin at root
        self.partial_path.append(root_f)

        # Recursively get all paths of left subtree
        self.get_paths_recur(left_node, 'left')

        # Recursively get all paths of right subtree
        self.get_paths_recur(right_node, 'right')

        # Return all paths and leaf labels of tree
        return (self.paths, self.preds)

    # Recursively evaluate data block to build path list
    def get_paths_recur(self, node, side):
        # Check if root node of subtree is a leaf node
        if self.d_block[(node, status)] == -1:
            self.paths.append(list(self.partial_path))

            # Append self to preds (list of leaf node class labels)
            self.preds.append(int(self.d_block[(node, pred)] - 1))

        # If not a leaf node, add to path and make recursive call
        else:

            # Append feature at self to path
            root_f = int(self.d_block[(node, split_f)] - 1)
            self.partial_path.append(root_f)

            # Access children nodes
            left_node = int(self.d_block[(node, l_node)] - 1)
            right_node = int(self.d_block[(node, r_node)] - 1)

            # Recursively call on children
            self.get_paths_recur(left_node, 'left')
            self.get_paths_recur(right_node, 'right')

        if side == 'right':
            self.partial_path = self.partial_path[:-1]

    # Recursively evaluate data block to build node list
    def get_nodes(self):

        # Build node object for root
        root_num = 0
        root_f = int(self.d_block[(root_num, split_f)] - 1)
        root_node = Node(root_num, root_f, 1, [root_f])

        # Build node object for left child
        left_node_num = int(self.d_block[(root_num, l_node)] - 1)
        left_f = int(self.d_block[(left_node_num, split_f)] - 1)
        left_node = Node(left_node_num, left_f, 2, [root_f, left_f])

        # Build node object for right child
        right_node_num = int(self.d_block[(root_num, r_node)] - 1)
        right_f = int(self.d_block[(right_node_num, split_f)] - 1)
        right_node = Node(right_node_num, right_f, 2, [root_f, right_f])

        # Set children of root
        root_node.set_children(left_node, right_node)

        # Append root to list of nodes
        self.nodes.append(root_node)

        # Make recursive call on children
        self.get_nodes_recur(left_node)
        self.get_nodes_recur(right_node)

        # Update class probabilisties of root (bubbles up from leaf nodes to root)
        root_node.update_probs()

        return self.nodes

    # Recursively evaluate data block to build node list
    def get_nodes_recur(self, node):

        # Check if root node of subtree is a leaf node
        if self.d_block[(node.num, status)] == -1:

            # If so, class prob is 100% for the class label
            c = int(self.d_block[(node.num, pred)])
            count = {c: 1.}
            node.count = count

        # If root node of subtree is not a leaf node
        else:

            # Append self to node list
            self.nodes.append(node)

            # Build node object for left child
            left_node_num = int(self.d_block[(node.num, l_node)] - 1)
            left_f = int(self.d_block[(left_node_num, split_f)] - 1)
            left_node = Node(left_node_num, left_f, node.path_len + 1, node.subpath+[left_f])

            # Build node object for right child
            right_node_num = int(self.d_block[(node.num, r_node)] - 1)
            right_f = int(self.d_block[(right_node_num, split_f)] - 1)
            right_node = Node(right_node_num, right_f, node.path_len + 1, node.subpath+[right_f])

            # Set children of root node of subtree
            node.set_children(left_node, right_node)

            # Make recursive call on children
            self.get_nodes_recur(left_node)
            self.get_nodes_recur(right_node)

            # Update probabilities of root node of subtree (bubbles up from leaf nodes)
            node.update_probs()

# Sanity check to make sure info matches tree structure written by R
def print_nodes(nodes):

    for node in nodes:
        print('---------------------------')
        print('Node number: ' + str(node.num))
        print('Feature: ' + str(node.feature))
        print('Children: ' + str(node.left_child.num) + ' & ' + str(node.right_child.num))
        print('Probabilities: ' + str(node.probs))
        print('Counts: ' + str(node.count))
        print('S: ' + str(node.path_len))

##########################################
# Read forest file format outputted by R #
##########################################
def read_trees(filename):

    # Initialize empty list of trees
    trees = []

    # Read filename written by R
    with open(filename, 'r') as (f):
        lines = f.readlines()

    # Append an empty array for first tree
    trees.append(np.zeros((0, len(lines[1].split(' ')) - 1)))

    # For each line of file
    for line in lines[1:]:

        # If an empty line, initialize an empty array for the next tree
        if line == '\n':
            trees.append(np.zeros((0, len(lines[1].split(' ')) - 1)))

        # If line has content, read its content and store as array
        else:
            arr = np.array([ line.split(' ')[i] for i in [0, 1, 2, 4, 5] ], dtype=int)
            trees[-1] = np.row_stack((trees[-1], arr))

    return trees

################################################
# Read variable importance file outputted by R #
################################################
def read_importances(filename):
    perm, gini = np.loadtxt(filename, skiprows=1, usecols=(-2,-1), unpack=True)
    return perm, gini

##################################################################
# Read class-specific permutation importance file outputted by R #
##################################################################
def read_class_importances(filename):
    data = np.loadtxt(filename, skiprows=1)
    cols = range(0, data.shape[1]-2)
    return data[:,cols]


