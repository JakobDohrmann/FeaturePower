from r2python import Tree, read_trees

trees = read_trees('./segmentation/single_run/forest.csv')

t = Tree(trees[0])

paths, c = t.get_paths()

for (path,pred) in zip(paths, c):
    print(f'path: {path}, prediction: {pred}')

