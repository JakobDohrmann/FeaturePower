
#####################################################
#   Feature power and coalition power computation   #
#####################################################

from itertools import *
from sklearn.datasets import *
from r2python import Tree
import gmpy


###############################################
# Compute pathPow in a tree for a given class #
###############################################
def eval_pathPow(paths, c, fnames, one_class):

    # Determine total number of features
    n = len(fnames)

    # Initialize empty list of feature powers
    power = []

    # Compute power for each feature
    for i,f in enumerate(fnames):

        # Determine root-to-leaf paths in which f is a member
        inclusions = [(path, v) for path,v in zip(paths,c) if f in path]

        # Determine root-to-leaf paths in which f is NOT a member
        exclusions = [(path, v) for path,v in zip(paths,c) if not f in path]

        inc_sum = 0
        exc_sum = 0

        # Compute first term of Shapley value
        for S,v in inclusions:

            # Determine length of path
            s = len(S)

            # Convert one_class from class label to characteristic function value (v)
            if (one_class == v+1):
                one_v = 1
            else:
                one_v = 0

            # Add to summation on first term
            inc_sum += gmpy.fac(s-1) * gmpy.fac(n-s) * one_v

        # Compute second term of Shapley value
        for S,v in exclusions:

            # Determine length of path
            s = len(S)

            # Convert one_class from class label to characterstic function value (v)
            if (one_class == v):
                one_v = 1
            else:
                one_v = 0

            # Add to summation on second term
            exc_sum += gmpy.fac(s) * gmpy.fac(n-s-1) * one_v

        # Compute difference of first and second term divided by n!
        power.append((inc_sum - exc_sum) / gmpy.fac(n))

    return power


#####################################################
# Compute strictNodePow in a tree for a given class #
#####################################################
def eval_strictNodePow(nodes, fnames, one_class):

    # Determine total number of features
    n = len(fnames)

    # Initialize empty dictionary for each feature:(inc,exc) mapping
    inc_exc = dict((f, (0., 0.)) for f in fnames)

    # Loop through nodes (i.e. all root-to-internal node subpaths)
    for node in nodes:

        # Determine length of subpath
        s = node.path_len

        # If class of interest is not already stored in probs, there are no
        # leaf nodes corresponding to this class, so the prob is 0
        if not one_class in node.probs:
            node.probs[one_class] = 0.

        # Compute a single inclusion term
        term = gmpy.fac(s-1) * gmpy.fac(n-s) * node.probs[one_class]

        # Add inclusion term to summation over all inclusions for last feature in subpath
        old_tuple = inc_exc[fnames[node.feature]]
        inc_exc[fnames[node.feature]] = (old_tuple[0] + term, old_tuple[1])

        # Count as an exclusion for all other features
        for f in fnames:
            if not f == fnames[node.feature]:

                # Add to summation over all exclusions for all other features
                # (including those in the subpath because strictNodePow method is strict)
                term = gmpy.fac(s) * gmpy.fac(n-s-1) * node.probs[one_class]
                inc_exc[f] = (inc_exc[f][0], inc_exc[f][1] + term)

    # Build feature power dictionary
    power = {}
    for f in fnames:

        # Compute the difference of inclusion and exclusion terms divided by n!
        power[f] = (inc_exc[f][0] - inc_exc[f][1]) / gmpy.fac(n)

    return power

###################################################
#  Compute cumNodePow in a tree for a given class #
###################################################
def eval_cumNodePow(nodes, fnames, one_class):

    # Determine total number of features
    n = len(fnames)

    # Initialize empty inclusion/exclusion dictionary
    inc_exc = {}

    # Compute feature power for each feature
    for i,f in enumerate(fnames):

        inc_exc[f] = (0., 0.)

        # Loop through nodes (i.e. root-to-internal node subpaths)
        for node in nodes:

            # Determine length of subpath
            s = node.path_len

            # If class of interest is not already stored in probs, there are no
            # leaf nodes corresponding to this class, so the prob is 0
            if not one_class in node.probs:
                node.probs[one_class] = 0.

            # Add to inclusion of that feature if the feature is in the subpath
            if i in node.subpath:
                term = gmpy.fac(s-1) * gmpy.fac(n-s) * node.probs[one_class]
                old_tuple = inc_exc[f]
                inc_exc[f] = (old_tuple[0] + term, old_tuple[1])

            # Add to the exclusion of that feature if the feature is not in the subpath
            else:
                term = gmpy.fac(s) * gmpy.fac(n-s-1) * node.probs[one_class]
                old_tuple = inc_exc[f]
                inc_exc[f] = (old_tuple[0], old_tuple[1] + term)

    # Build feature power dictionary as the difference of inclusions and exclusions divided by n!
    power = {}
    for f in fnames:
        power[f] = (inc_exc[f][0] - inc_exc[f][1]) / gmpy.fac(n)

    return power

###########################################
#    Compute count importance in a RF     #
###########################################
def count_importance(trees, fnames, cnames):

    # Initialize empty count dictionary
    counts = dict((f, 0) for f in fnames)

    # Count feature nodes in each tree
    for tree in trees:

        # Access nodes in tree
        t = Tree(tree)
        nodes = t.get_nodes()

        # Increment count of each feature when encountered in node
        for node in nodes:
            counts[fnames[node.feature]] += 1

    # Rank features by count
    rank = [f for f,v in sorted(counts.items(), reverse=True, key=lambda x:x[1])]

    return rank, counts

################################################################
# Compute any type of feature power in a RF for a given class  #
################################################################
def feature_power(trees, fnames, cnames, one_class, method):

    if method == 'path':
        return feature_power_path(trees, fnames, cnames, one_class)
    elif method == 'strictNode' or method == 'cumNode':
        return feature_power_node(trees, fnames, one_class, method)
    else:
        print('Method ' + str(method) + ' not supported')

###################################################
#  Compute pathPow in entire RF for a given class #
###################################################
def feature_power_path(trees, fnames, cnames, one_class):

    # Initialize tree-specific power list
    separated_power = []

    # Initialize forest-wide power list
    aggregate_power = len(fnames)*[0]

    # Compute pathPow for each tree
    for tree in trees:

        # Access paths of decision tree
        t = Tree(tree)
        paths, c = t.get_paths()

        # Evaluate pathPow for tree
        separated_power.append(eval_pathPow(paths, c, fnames, one_class))

        # Add to forest-wide power evaluation
        for i in range(len(fnames)):
            aggregate_power[i] += separated_power[-1][i]

    # Average power over forest
    for i in range(len(fnames)):
        aggregate_power[i] = aggregate_power[i] / len(trees)

    # Rank features high to low by importance
    rank = [fnames[i] for i,v in sorted(enumerate(aggregate_power), reverse=True, key=lambda x:x[1])]
    power = dict((fnames[i],p) for i,p in enumerate(aggregate_power))

    return rank, power

#############################################################
#  Compute cum/strictNodePow in entire RF for a given class #
#############################################################
def feature_power_node(trees, fnames, one_class, method):

    # Initialize tree-specific power list
    separated_power = []

    # Initialize forest-wide power dictionary
    power = dict((f, 0.) for f in fnames)

    # Compute cumNodePow or strictNodePow for each tree (depending on method param)
    for tree in trees:

        # Access paths in decision tree
        t = Tree(tree)
        nodes = t.get_nodes()

        # Evaluate strictNodePow or pathNodePow within tree
        if method == 'strictNode':
            separated_power.append(eval_strictNodePow(nodes, fnames, one_class))
        elif method == 'cumNode':
            separated_power.append(eval_cumNodePow(nodes, fnames, one_class))

        # Add to forest-wide power evaluation
        for f in separated_power[-1].keys():
            power[f] += separated_power[-1][f]

    # Average power over forest
    for f in fnames:
        power[f] = power[f] / len(trees)

    # Rank features high to low by power
    rank = [f for f,v in sorted(power.items(), reverse=True, key=lambda x:x[1])]

    return rank, power

# Build powerset of items (helper for coalition power)
def powerset(iterable):
    s = set(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(2,len(s)+1)))

###########################################################
#  Compute coalition power in entire RF for a given class #
###########################################################
def coalition_power(trees, fnames, cnames, one_class, method):

    # Initialize empty forest-wide power list
    power = []

    # Build all possible pairs of features 1-8 (only works on toys data set)
    coalitions = powerset(range(8))

    # Compute coalition power for each coalition
    for i,coalition in enumerate(coalitions):

        coalition = set(coalition)

        # Evaluate power according to method param
        if method == 'path':
            power.append(coalition_power_path(trees, fnames, one_class, coalition))
        elif method == 'cumNode':
            power.append(coalition_power_cumNode(trees, fnames, one_class, coalition))
        else:
            print('Method ' + str(method) + ' not supported')
            return

    return power, coalitions

#########################################
#  Compute coalition count in entire RF #
#########################################
def coalition_count(trees):

    # Initialize empty forest-wide power list
    power = []

    # Build all possible pairs of features 1-8 (only works on toys data set)
    coalitions = powerset(range(8))

    # Compute coalition count for each coalition
    for i,coalition in enumerate(coalitions):
        coalition = set(coalition)
        power.append(count_coalitions(trees, coalition))

    return power, coalitions

###############################################################
#  Compute coalition count wrt one coalition in entire forest #
###############################################################
def count_coalitions(trees, coalition):

    count = 0

    # Loop over every tree
    for tree in trees:

        # Access all nodes in tree
        t = Tree(tree)
        nodes = t.get_nodes()

        # If a node is in the coalition of interest, increment
        for node in nodes:
            if coalition.issubset(node.subpath):
                count += 1

    return count

########################################################################
#  Compute pathPow-based coalition power in entire RF for a given class#
########################################################################
def coalition_power_path(trees, fnames, one_class, coalition):

    pow = 0.

    # Computute coalition power based on pathPow for each tree
    for tree in trees:

        # Access all paths and leaf labels of tree
        t = Tree(tree)
        paths, c = t.get_paths()

        # Evaluate power of coalitions using pathPow
        pow += eval_coalitions_path(paths, c, fnames, coalition, one_class)

    # Average power results over forest
    pow = pow / float(len(trees))

    return pow

#####################################################################
# Compute pathPow-based coalition power in a tree for a given class #
#####################################################################
def eval_coalitions_path(paths, c, fnames, coalition, one_class):

    # Determine total number of features
    n = len(fnames)

    # Determine root-to-leaf paths in which all members of coalition are included
    inclusions = [(path, v) for path,v in zip(paths,c) if coalition.issubset(path)]

    # Determine root-to-leaf paths in which all members of coalition are excluded
    exclusions = [(path, v) for path,v in zip(paths,c) if not coalition.issubset(path)]

    inc_sum = 0
    exc_sum = 0

    # Compute first term of Shapley value
    for S,v in inclusions:

        # Determine length of path
        s = len(S)

        # Convert one_class from class label to characteristic function value (v)
        if (one_class == v+1):
            one_v = 1
        else:
            one_v = 0

        # Add to summation on first term
        inc_sum += gmpy.fac(s-1) * gmpy.fac(n-s) * one_v

    # Compute second term of Shapley value
    for S,v in exclusions:

        # Determine length of path
        s = len(S)

        # Convert one_class from class label to characteristic function value (v)
        if (one_class == v):
            one_v = 1
        else:
            one_v = 0

        # Add to summation on second term
        exc_sum += gmpy.fac(s) * gmpy.fac(n-s-1) * one_v

    # Return power as the difference of two terms divided by n!
    return (inc_sum - exc_sum) / gmpy.fac(n)

########################################################################
# Compute cumNodePow-based coalition power in a tree for a given class #
########################################################################
def coalition_power_cumNode(trees, fnames, one_class, coalition):

    pow = 0.

    # Computute coalition power based on cumNodePow for each tree
    for tree in trees:

        # Access all paths and leaf labels of tree
        t = Tree(tree)
        nodes = t.get_nodes()

        # Evaluate power of coalitions using cumNodePow
        pow += eval_coalitions_cumNode(nodes, fnames, coalition, one_class)

    # Average power results over forest
    pow = pow / float(len(trees))

    return pow

########################################################################
# Compute cumNodePow-based coalition power in a tree for a given class #
########################################################################
def eval_coalitions_cumNode(nodes, fnames, coalition, one_class):

    # Determine total number of features
    n = len(fnames)

    inclusions = []
    exclusions = []

    # Iterate over every node (i.e. every root-to-internal node subpath)
    for node in nodes:

        # If class of interest is not already stored in probs, there are no
        # leaf nodes corresponding to this class, so the prob is 0
        if not one_class in node.probs:
            node.probs[one_class] = 0.

        # Determine root-to-leaf paths in which all members of coalition are included
        if coalition.issubset(node.subpath):
            inclusions.append((node.subpath, node.probs[one_class]))

        # Determine root-to-leaf paths in which all members of coalition are excluded
        else:
            exclusions.append((node.subpath, node.probs[one_class]))

    inc_sum = 0.
    exc_sum = 0.

    # Compute first term of Shapley value
    for S,v in inclusions:
        s = len(S)
        inc_sum += gmpy.fac(s-1) * gmpy.fac(n-s) * v

    # Compute second term of Shapley value
    for S,v in exclusions:
        s = len(S)
        exc_sum += gmpy.fac(s) * gmpy.fac(n-s-1) * v

    # Calculate power as the difference of both terms divided by n!
    return (inc_sum - exc_sum) / gmpy.fac(n)
