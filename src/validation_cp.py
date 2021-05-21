
#################################################
# Obtain results of coalition power computation #
#################################################

from extract_rules import coalition_power
from extract_rules import coalition_count
import numpy as np
from r2python import read_trees 
import subprocess 
import argparse 

###########################################################################
# Write importance values obtained by coalition power and coalition count #
###########################################################################
def write_importances(importances, coalitions, measures, f_names, datadir):

    # Determine number of RF instances created  
    n_iter = len(importances[measures[-1]]) 

    # Write results of coalition power evaluation to file
    for m,measure in enumerate(measures):

        # Determine number of coalitions evaluated
	n_coalitions = len(coalitions[measure][0])

	x = []
	y = np.zeros(n_iter*n_coalitions)

        # For every RF instance created, add brackets around the set for file formatting
	for j in range(n_iter):
            for i,coalition in enumerate(coalitions[measure][j]):
		f_coalition = '{'
		for c in coalition:
                    f_coalition += str(f_names[c])
                    if not c == coalition[-1]:
			f_coalition += ', '
		f_coalition += '}'
                
                # Access associated info about coalition
		x.append(f_coalition)
		y[(j*n_coalitions)+i] = importances[measure][j][i]

        # Write results for each measure to file
	with open(datadir+measure+'.txt', 'w') as f:
            for x1,y1 in zip(x, y):
		f.write(str(x1) + '\t' + str(y1) + '\n')

def main(args):

    datadir = args.data + '/'

    # Determine the method(s) specified by arguments
    if args.powMethod == 'all':
	methods = ['path', 'cumNode']
    else:
        methods = [args.powMethod]

    # If a ntree range is specified, increment by 50
    if args.ntreeMin > 0 and args.ntreeMax > 0:
        ntrees = [5]
        ntrees += range(args.ntreeMin, args.ntreeMax+1, 50)
    else:
        ntrees = [args.ntree]
    
    # If a mtry range is specified, increment by 2
    if args.mtryMin > 0 and args.mtryMax > 0: 
        mtrys = range(args.mtryMin, args.mtryMax+1, 2)
    else:
        mtrys = [args.mtry]

    n_iter = args.iter

    # Hard code feature names and class label names by data set
    if args.data == 'toys':
        feature_names = range(1, 201)
        class_names = (1, 2)
    elif args.data == 'wdbc':
        feature_names = range(1, 31)
        class_names = (1, 2)
    elif args.data == 'wine':
        feature_names = range(1,14)
        class_names = range(1, 4)
    elif args.data == 'iris':
        feature_names = range(1,5)
        class_names = range(1,4)
    elif args.data == 'segmentation':
            feature_names = range(1, 20)
        class_names = range(1,8) 
    else:
        print('Update dataset feature and class names!!!!')

    # Set file names for every measure 
    measures = []
    for method in methods:
        for c in class_names:
            measures.append(method + ' coalition power for class ' + str(c))
    measures.append('Coalition count')
    
    # Train forest and evaluate coalition power for each hyperparameter
    for ntree in ntrees:
        for mtry in mtrys:

            # Initialize empty dictionary for coalition power ranking
            coalitions = {}
            for method in methods:
                for c in class_names:
                    coalitions[method + ' coalition power for class ' + str(c)] = []
            coalitions['Coalition count'] = []

            # Initialize empty dictionary for coalition power
            importances = {}
            for method in methods:
                for c in class_names:
                    importances[method + ' coalition power for class ' + str(c)] = []
            importances['Coalition count'] = []

            # Print hyperparameters being used
            print
            print(ntree)
            print(mtry)
            print

            # Initialize empty matrix of pairwise rank-order correlations between measures
            spearmans_rho = np.zeros((n_iter, len(importances.keys()), len(importances.keys())))

            # Train many (n_iter) instances of RF 
            for iter in range(n_iter):

                # Train a forest in R
                retcode = subprocess.call("/usr/bin/Rscript --vanilla -e 'source(\"train_forest.r\")' " + str(ntree) + ' ' + str(mtry) + ' ' + args.data + ' ' + str(len(class_names)+2), shell=True)

                # Read R results in
                trees = read_trees(datadir+'forest.csv')

                # Compute class-specific coalition power for each method specified 
                # (pathPow, cumNodePow, strictNodePow)
                for ci,c in enumerate(class_names):
                    for method in methods:
                        pow, co = coalition_power(trees, feature_names, class_names, c, method)
                        measure = method+' coalition power for class ' + str(c)
                        coalitions[measure].append(co)
                        importances[measure].append(pow)

                # Compute coalition count for RF
                pow, co = coalition_count(trees)
                measure = 'Coalition count'
                coalitions[measure].append(co)
                importances[measure].append(pow)

                # Write importances to file
                write_importances(importances, coalitions, measures, feature_names, datadir)

if __name__=='__main__':

	# Set up arg parser 
	helpstr = 'Train multiple random forest instances and compare coalition count to coalition power'
	parser = argparse.ArgumentParser(description=helpstr);
	parser.add_argument('-n', '--ntree', dest='ntree', type=int, help='number of trees', default=50)
	parser.add_argument('-m', '--mtry', dest='mtry', type=int, help='number of candidate features', default=66)
	parser.add_argument('-i', '--iter', dest='iter', type=int, help='number of random iterations', default=50)
	parser.add_argument('-a', '--ntreeMin', dest='ntreeMin', type=int, help='min of range for ntree search', default=-1)
	parser.add_argument('-b', '--ntreeMax', dest='ntreeMax', type=int, help='max of range for ntree search', default=-1)
	parser.add_argument('-c', '--mtryMin', dest='mtryMin', type=int, help='min of range for mtry search', default=-1)
	parser.add_argument('-d', '--mtryMax', dest='mtryMax', type=int, help='max of range for mtry search', default=-1)
	parser.add_argument('-p', '--powMethod', dest='powMethod', type=str, help='method for feature power (options are path, strictNode, or cumNode)', default='all')

	parser.add_argument('-f', '--data', dest='data', type=str, help='folder name of dataset', default='toys')
	parser.add_argument('-t', '--top', dest='top', type=int, help='num of top ranked features', default=7)
	
	args = parser.parse_args()

	main(args)
