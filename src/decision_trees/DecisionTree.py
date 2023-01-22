import numpy as np
from Node import *


class DecisionTreeClassifier:
    def __init__(self, min_samples_left=2, max_tree_depth=2):
        # initialize the root used to move through the tree
        self.root = None
        # stopping conditions
        # stop splitting if there are less than ... samples
        self.min_samples_left = min_samples_left
        # stop splitting if the tree has as many split levels as ...
        self.max_tree_depth = max_tree_depth
    def create_tree(self, dataset, current_depth=0):
        # slicing - take every row and every column except the last one
        features = dataset[:, :-1]
        # slicing - take every row and the last column
        assignation = dataset[:, -1]

        # number of rows in the feature part of the dataset
        num_samples = len(features)
        # number of columns -""-
        num_features = len(features[0])

        # if no stopping conditions are met
        if num_samples >= self.min_samples_left and current_depth <= self.max_tree_depth:
            current_depth=current_depth+1
            # split the tree into two datasets
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # recursively generate tree branches as next trees
            leftbranch = self.create_tree(best_split[0], current_depth)
            rightbranch = self.create_tree(best_split[1], current_depth)
            # return a decision node with the index of the feature, the point used as the splitting value, left and right branches and the information gain
            return Node(best_split[2], best_split[3], leftbranch, rightbranch, best_split[4])
        else:
            # compute leaf node
            leaf_value = self.calc_leaf_value(assignation)
            # return a leaf node
            return Node(value=leaf_value)


    def get_best_split(self,dataset, num_samples, num_features):
            alldatapt = []
            maxinfogain = -1
            bestsplits=tuple()

            for i in range (num_features):#columns
                for j in range (num_samples):#rows
                    # collecting all datapoints from this column (one feature)
                    alldatapt.append(dataset[j,i])
                # deleting repeats
                uniquedatapt = np.unique(alldatapt)

                for uniquepoint in uniquedatapt:
                    leftsplit = np.array([row for row in dataset if row[i] <= uniquepoint])
                    rightsplit = np.array([row for row in dataset if row[i] > uniquepoint])

                    if len(leftsplit)>0 and len(rightsplit)>0:
                        y, left_y, right_y = dataset[:, -1], leftsplit[:, -1], rightsplit[:, -1]
                        tempinfogain=self.calc_infogain(y,left_y,right_y)
                        if tempinfogain > maxinfogain:
                            maxinfogain = tempinfogain
                            #left dataset, right dataset, index of feature used to split, point used to split, information gained by using this split
                            bestsplits = (leftsplit, rightsplit, i, uniquepoint, maxinfogain)

            return bestsplits




    def calc_infogain(self,alldatapt, lsplit, rsplit):
        weightleft = len(lsplit)/len(alldatapt)
        weightright = len(rsplit)/len(alldatapt)
        infogain = self.calc_gini(alldatapt)-(weightleft*self.calc_gini(lsplit)+weightright*self.calc_gini(rsplit))

        #infogain = self.calc_entropy(alldatapt)-(weightleft*self.calc_entropy(lsplit)+weightright*self.calc_entropy(rsplit))

        return infogain

    def calc_entropy(self,val):
        assignation_types=np.unique(val)
        entropy=0
        for i in assignation_types:
            assignation_num = 0
            for j in val:
                if j == i:
                    assignation_num = assignation_num+1

            entropy=entropy-(assignation_num/len(val))*np.log2(assignation_num/len(val))
        return entropy

    def calc_gini(self,val):
        assignation_types = np.unique(val)
        gini=1
        for i in assignation_types:
            assignation_num=0
            for j in val:
                if j==i:
                    assignation_num = assignation_num + 1

            gini=gini-(assignation_num/len(val))**2
        return gini

    def calc_leaf_value(self, val):

        val = list(val)
        return max(val, key=val.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)



    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.create_tree(dataset)

    def predict(self, X):

        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)