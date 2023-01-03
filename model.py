import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifiers():
    def __init__(self, bootstrap=True, n_estimators=100, max_depth=10,
                 min_samples_leaf=1, min_samples_split=2, max_features='sqrt'):
        # A random forest classifier is a estimator that fits a
        # number of decision tree classifiers on various sub-samples
        # mof the dataset.

        self.bootstrap = bootstrap
        self.max_features = max_features
        # The sub-sample size is controlled with the `max_samples` parameter if
        # `bootstrap=True` (default), otherwise the whole dataset is used to build
        # each tree.

        self.n_estimators = n_estimators
        # The number of trees in the forest.

        self.max_depth = max_depth
        # The maximum depth of the tree.

        self.min_samples_leaf = min_samples_leaf
        # The minimum number of samples required to be at a leaf node.
        # A split point at any depth will only be considered if it leaves at
        # least ``min_samples_leaf`` training samples in each of the left and
        # right branches.

        self.min_samples_split = min_samples_split
        # The minimum number of samples required to split an internal node

    def fit(self, X, y):
        # Build a forest of trees from the training set (X, y).
        self.X, self.y = X, y
        self.trees = [self.create_tree() for i in range(self.n_estimators)]

    def create_tree(self):
        # A function to create sub-dataset from the original dataset and to build
        # tree based on it.

        if self.bootstrap:
            random_idx = np.random.choice(len(self.y), len(self.y))
        else:
            random_idx = np.random.choice(
                len(self.y), len(self.y), replace=False)
        # Whether bootstrap samples are used when building trees. If False, the
        # whole dataset is used to build each tree.

        if self.max_features == 'sqrt':
            max_cols = int(np.sqrt(self.X.shape[1]))
        else:
            max_cols = self.X.shape[1]
        random_col = np.random.choice(self.X.shape[1], max_cols, replace=False)
        # The number of features to build trees and to look for the best split:
        #  - If "sqrt", then `max_features=sqrt(n_features)`.
        #  - If None, then `max_features=n_features`.

        return DecisionTreeClassifiers(X=self.X.iloc[random_idx, random_col], y=self.y[random_idx],
                                       idxs=np.arange(len(self.y)), max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split)
        # Building a tree with random samples data and features from the original dataset.

    def predict(self, data):
        # A function to make presictions using majority voting system.
        # The predicted class of an input sample is a vote by the trees in
        # the forest. The predicted class is decided by comparing
        # their probability estimates with a threshold (default is 0.5)

        predictions = sum([tree.predict(data)
                          for tree in self.trees]) / self.n_estimators
        # Probability estimates are calculated by total predictions made
        # by all the trees divided by total of trees.

        return np.where(predictions <= 0.5, 0, 1)
        # Final predictions, a sample is belong to class 0 if it's
        # probability estimate is less or equal to 0.5 and vice versa.


class DecisionTreeClassifiers():
    def __init__(self, X, y, idxs, max_depth=10, min_samples_leaf=1, min_samples_split=2):
        self.max_depth, self.idxs, self.col_idx = max_depth, idxs, float('inf')
        # - idxs are indexes to keep track of which row indexes went to left-hand
        # side of the tree, which went to right-hand side of the tree when splitting
        # tree into sub trees.
        # - col_idx is a index of which feature that split tree with a lowest gini
        # score. col_idx = infinite when initializing and = -1 when tree can not
        # find a feature to split node.

        self.X, self.y = X.iloc[self.idxs, :], y.iloc[self.idxs]
        self.origin_X, self.origin_y = X, y
        # X and y are training data which located by idxs from each node.
        # original_X and original_y are original training data .

        self.min_samples_leaf, self.min_samples_split = min_samples_leaf, min_samples_split
        self.row, self.column = len(self.y), X.shape[1]
        # Number of samples and features of a node.

        self.values = [np.count_nonzero(
            self.y.values == 0), np.count_nonzero(self.y.values == 1)]
        # Number of samples in each classes.

        self.score = self.gini_score(self.values)
        # Gini score of a node.

        if self.is_leaf:
            # Stop splitting and return to the previous node.
            return

        self.find_varsplit()
        # Find feature to split a tree into sub-trees.

    @property
    def is_leaf(self):
        # Check if a node is a leaf or not. A node is a leaf node when it has less samples
        # than min_samples_split or it's purify (gini score = 0.0) or it reachs max depth.
        return self.row < self.min_samples_split or self.score == 0.0 or self.max_depth == 0

    def find_varsplit(self):
        # A function to find a feature to split the tree.

        for i in range(self.column):
            self.find_best_split(i)
        # Go through each feature and find the best one to split the tree.

        if self.col_idx == float('inf'):
            self.col_idx = -1
            return
        # Can not find a feature to split so 'col_idx' equals to -1 and return to previous
        # node.

        col_var = self.X.iloc[:, self.col_idx].values
        left_idxs = np.nonzero(col_var <= self.varsplit)[0]
        right_idxs = np.nonzero(col_var > self.varsplit)[0]
        # Found a feaure to split, take all the indexes belong to left-side node and all the
        # indexes belong to right-side note based on 'varsplit' and continue split the left and
        # right node.

        self.left_node = DecisionTreeClassifiers(X=self.origin_X, y=self.origin_y, idxs=self.idxs[left_idxs],
                                                 max_depth=self.max_depth - 1, min_samples_leaf=self.min_samples_leaf,
                                                 min_samples_split=self.min_samples_split)
        self.right_node = DecisionTreeClassifiers(X=self.origin_X, y=self.origin_y, idxs=self.idxs[right_idxs],
                                                  max_depth=self.max_depth - 1, min_samples_leaf=self.min_samples_leaf,
                                                  min_samples_split=self.min_samples_split)
        # Split the left and right node, 'max_depth' decreased by 1.

    def gini_score(self, values):
        # A function to calculate gini score for a node. A gini score for a node is
        # 1 minus the probability of class 1 squared minus the probability of class
        # 0 squared
        return 1 - (1 - values[0] / sum(values))**2 - (1 - values[1] / sum(values))**2

    def find_best_split(self, feat_idx):
        # A function to find the best feature to split the tree.

        X, y = self.X.iloc[:, feat_idx], self.y
        # Take all the values from a feature by feat_idx.

        sorted_df = pd.concat([X, y], axis=1).sort_values(
            by=self.X.columns[feat_idx])
        sorted_X, sorted_y = sorted_df.iloc[:, 0], sorted_df.iloc[:, 1]
        right_samples, right_values, right_gini = self.row, [
            self.values[0], self.values[1]], self.score
        left_samples, left_values, left_gini = 0, [0, 0], float('inf')
        # Sort data based on X values, take all these samples to the right-side node so
        # 'right_samples', 'right_values' will have all the samples, number of values in
        # each class then calculate the gini score. The left-side note has nothing in the
        # start so 'left_samples', 'left_values' equal to zeros so it doesn't has gini score.

        for i in range(0, self.row - self.min_samples_leaf):
            # Iterate each row in X, tranfer one value from right-side note to left-side note.

            Xi, yi = sorted_X.iloc[i], sorted_y.iloc[i]
            split = (Xi + sorted_X.iloc[i + 1]) / 2
            # 'split' is a value to split all the samples, if smaller than it then go to left-side
            # and vice versa. 'split' calculated by adding two consecutive values as a threshod to
            # split the tree and make predictions.

            left_samples += 1
            right_samples -= 1
            if yi == 0:
                left_values[0] += 1
                right_values[0] -= 1
            else:
                left_values[1] += 1
                right_values[1] -= 1
            # update all the variables.

            if (i < self.min_samples_leaf - 1) or (Xi == sorted_X.iloc[i + 1] and Xi == sorted_X.iloc[i + 2]):
                continue
            # Each node must have 'min_samples_leaf' sample(s) and three consecutive values must
            # difference from each other or the threshold will be the same.

            right_gini = self.gini_score(right_values)
            left_gini = self.gini_score(left_values)
            right_pos = sum(right_values) / (right_samples + left_samples)
            left_pos = sum(left_values) / (right_samples + left_samples)
            split_gini = left_gini * left_pos + right_gini * right_pos
            # Calculate gini score for both side and total gini score for the feature to split node.
            # The total gini score is the weighted average of the leaf gini score. The weighted average
            # is the total number of samples divided by the total number of samples in both leaves.
            # The smaller total gini score, the better that feature for splitting the tree.

            if split_gini < self.score:
                self.col_idx, self.colsplit = feat_idx, self.X.columns[feat_idx]
                self.score, self.varsplit = split_gini, split
            # Update if found a better gini score, 'colsplit' is the name of the feature to split.

    def predict(self, data):
        # A function to make prediction for each samples in dataset.
        return np.array([self.predict_row(i[1]) for i in data.iterrows()])

    def predict_row(self, row):
        # A function to make prediction for a single sample.
        if self.is_leaf:
            return self.predict_class()
        # If the current node is a leaf then classify the sample.

        goto = self.left_node if row[self.colsplit] <= self.varsplit else self.right_node
        # If not the leaf node then go to left-side node if with the
        # splitting feature in the current node, the sample's value is
        # less or equal to the splitting value or vice versa.

        return goto.predict_row(row)

    def predict_class(self):
        # A function to classify a sample. A sample is belong to
        # class 0 if the total values of class 0 is greater or equal
        # to the total values of class 1 in the leaf node which sample
        # fall in.
        return 0 if self.values[0] >= self.values[1] else 1
