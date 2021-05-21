
###############################################
# Obtain results of feature power computation #
###############################################

from extract_rules import feature_power
from extract_rules import count_importance
import numpy as np
from scipy.stats import spearmanr
from r2python import read_trees
from r2python import read_importances
from r2python import read_class_importances
import subprocess
import argparse

#####################################
# Compute rank-order Spearman's rho #
#####################################
def pairwise_spearman(rank1, rank2):
    corr, p = spearmanr(rank1, rank2)
    return corr

###########################################################################
# Write importance values obtained by feature power and other VI measures #
###########################################################################
def write_importances(importances, measures, f_names, datadir):

    # Determine total number of features
    n_features = len(f_names)

    # Determine number of RF instances created
    n_iter = len(importances[measures[-1]])

    # Save each importance value under the appropriate measure file name
    for measure in measures:
        x = []
        y = np.zeros(n_iter*n_features)
        for j in range(n_iter):
            for i,f in enumerate(f_names):
                x.append(f)
                y[(j*n_features)+i] = importances[measure][j][f]
        np.savetxt(datadir+measure+'.txt', np.column_stack((x, y)))

################################################################################
# Write mean std in importance values from feature power and other VI measures #
################################################################################
def write_mean_std(importances, measures, append, datadir):

    # Compute std for each VI measure (and FP) and save to text file
    for measure in measures:
        n_iter = len(importances[measures[-1]])
        tmp = np.zeros((6, n_iter))
        for i in range(n_iter):
            tmp[:,i] = np.array(list(importances[measure][i].values())[:6],dtype=np.float64)

        imp = (np.std(tmp, axis=1) - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        with open(datadir+'meanStd'+measure+'.txt', append) as f:
            f.write(str(np.mean(imp)) + '\n')

#######################################################
# Write method rank-order correlation results to file #
#######################################################
def write_spear_rho(mean, std, methods, datadir):

    # Create header based on method name
    header = methods[0].replace('Node','') + '1\t' + methods[0].replace('Node','') + '2'
    for method in methods[1:]:
        header += '\t' + method.replace('Node','') + '1\t' + method.replace('Node','') + '2'
    header += '\tPerm1\tPerm2\tGini\tPerm\tCount\n\n'

    # Write pairwise correaltion between methods to file
    with open(datadir+'correlation_results.txt', 'w') as f:
        f.write(header)
        for row in mean:
            tmp = str(row[0])
            for x in row[1:]:
                tmp += '\t' + str(x)
            f.write(tmp + '\n')
        f.write('\n')
        for row in std:
            tmp = str(row[0])
            for x in row[1:]:
                tmp += '\t' + str(x)
            f.write(tmp + '\n')

##########################################################
# Write stability of feature power and other VI measures #
##########################################################
def write_stability_spear_rho(ranks, measures, append, datadir):
    # Write mean rank-order correlation of top 7 ranked features for each measure
    # this is often referred to as stability over random RF instances
    for measure in measures:
        # Compute every pairwise spearman's rho
        spear_sum = 0.0
        for i,rank1 in enumerate(ranks[measure]):
            for j,rank2 in enumerate(ranks[measure]):
                spear_sum += pairwise_spearman(rank1[:7], rank2[:7])

        # Write results to file
        with open(datadir+'meanCorr'+measure+'.txt', append) as f:
            if (i*j) == 0:
                f.write(str(spear_sum) + '\n')
            else:
                f.write(str(spear_sum / float(i*j)) + '\n')

######################################
# Main method to run all experiments #
######################################
def main(args):

    datadir = args.data + '/'

    # Clear accuracy file
    with open(datadir+'accuracy.csv', 'w') as f:
        f.write('')

    # Determine the method(s) specified by arguments
    if args.powMethod == 'all':
        methods = ['path', 'strictNode', 'cumNode']
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
            measures.append(method + 'Pow' + str(c))
    for c in class_names:
        measures.append('Perm' + str(c))
    measures.append('Gini')
    measures.append('Permutation')
    measures.append('Count')

    # Train forest and evaluate variable importance and feature power for each hyperparameter
    param_iter = 0
    for ntree in ntrees:
        for mtry in mtrys:

            # Initialize empty dictionary to store feature ranks from each VI/FP measure
            ranks = {}
            for method in methods:
                for c in class_names:
                    ranks[method + 'Pow' + str(c)] = []
            for c in class_names:
                ranks['Perm' + str(c)] = []
            ranks['Gini'] = []
            ranks['Permutation'] = []
            ranks['Count'] = []

            # Initialize empty dictionary to store feature importances from each VI/FP measure
            importances = {}
            for method in methods:
                for c in class_names:
                    importances[method + 'Pow' + str(c)] = []
            for c in class_names:
                importances['Perm' + str(c)] = []
            importances['Gini'] = []
            importances['Permutation'] = []
            importances['Count'] = []

            # Initialize empty matrix of pairwise rank-order correlations between measures
            spearmans_rho = np.zeros((n_iter, len(importances.keys()), len(importances.keys())))

            # Print hyperparameters being used
            print
            print('ntrees: ' + str(ntree))
            print('mtry: ' + str(mtry))
            print

            # Train many (n_iter) instances of RF
            for iter in range(n_iter):

                # Train a forest in R
                retcode = subprocess.call("/usr/bin/env Rscript --vanilla -e 'source(\"train_forest.r\")' " + str(ntree) + ' ' + str(mtry) + ' ' + args.data + ' ' + str(len(class_names)+2), shell=True)

                # Read R results in
                trees = read_trees(datadir+'forest.csv')

                # Read class-specific permutation importance from R
                class_perm = read_class_importances(datadir+'importances.csv')

                # Compute class-specific feature power for each method specified
                # (pathPow, cumNodePow, strictNodePow)
                for ci,c in enumerate(class_names):
                    for method in methods:
                        rank, pow = feature_power(trees, feature_names, class_names, c, method)
                        measure = method + 'Pow' + str(c)
                        ranks[measure].append(rank)
                        importances[measure].append(pow)

                        # Print type of feature power method and feature ranking
                        print(measure)
                        print(rank)

                    # Order features by class-specific permutation importance
                    rank = [feature_names[i] for i,v in sorted(enumerate(class_perm[:, ci]), reverse=True, key=lambda x:x[1])]
                    imp = dict((feature_names[i],p) for i,p in enumerate(class_perm[:, ci]))
                    measure = 'Perm' + str(c)
                    ranks[measure].append(rank)
                    importances[measure].append(imp)

                    # Print class-specific perm imp and feature ranking
                    print(measure)
                    print(rank)

                # Read class-aggregated permutation importance and Gini importance from R
                perm, gini = read_importances(datadir+'importances.csv')

                # Order features by Gini importance
                rank = [feature_names[i] for i,v in sorted(enumerate(gini), reverse=True, key=lambda x:x[1])]
                imp = dict((feature_names[i],p) for i,p in enumerate(gini))
                measure = 'Gini'
                ranks[measure].append(rank)
                importances[measure].append(imp)

                # Print Gini importance and feature ranking
                print(measure)
                print(rank)

                # Order features by permutation importance
                rank = [feature_names[i] for i,v in sorted(enumerate(perm), reverse=True, key=lambda x:x[1])]
                imp = dict((feature_names[i],p) for i,p in enumerate(perm))
                measure = 'Permutation'
                ranks[measure].append(rank)
                importances[measure].append(imp)

                # Print permutation importance and feature ranking
                print(measure)
                print(rank)

                # Compute count importance
                rank, imp = count_importance(trees, feature_names, class_names)
                measure = 'Count'
                ranks[measure].append(rank)
                importances[measure].append(imp)

                # Print count importance and feature ranking
                print(measure)
                print(rank)

                # Calculate pairwise rank-order correlation between methods of top args.top features
                for j,m1 in enumerate(measures[:-1]):
                    for k,m2 in enumerate(measures[j:]):
                        if k == 0:
                            spearmans_rho[iter, j, j] = 1.
                        else:
                            spearmans_rho[iter, j, k+j] = pairwise_spearman(ranks[m1][-1][:args.top], ranks[m2][-1][:args.top])
                spearmans_rho[iter, -1, -1] = 1.
            mean_spear_rho = np.mean(spearmans_rho, axis=0)
            std_spear_rho = np.std(spearmans_rho, axis=0)

            # Print Spearman's rho results
            print(measures)
            print(mean_spear_rho)
            print(std_spear_rho)

            # On first iteration for every hyperparameter, write files
            if param_iter == 0:
                write_spear_rho(mean_spear_rho, std_spear_rho, methods, datadir)
                write_importances(importances, measures, feature_names, datadir)
                write_mean_std(importances, measures, 'w', datadir)
                write_stability_spear_rho(ranks, measures, 'w', datadir)

            # On all other iterations, append to existing files
            else:
                write_mean_std(importances, measures, 'a', datadir)
                write_stability_spear_rho(ranks, measures, 'a', datadir)

            param_iter += 1

if __name__=='__main__':

    # Set up arg parser
    helpstr = 'Train multiple random forest instances and compare VI measurements to feature power'
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
