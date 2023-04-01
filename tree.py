from typing import List, Tuple
from summary import accuracy, histogram, confusion, print_confusion_matrix
from functools import reduce
from math import log2, inf
import json
from utility import fst, snd, idx_of_max, load_data, export_to_json, in_bound, fmap, get_col
import time

# structure of node 
# either:
# BRANCH ==> split_rule: { feature_idx, bounds }, children: Node[] (where idx is bound idx)
# or:
# LEAF ==> classification (letter)


# TYPES
class SplitRule:
    def __init__(self, feature_idx: int, bounds: Tuple[float, float]):
        self.feature_idx = feature_idx
        self.bounds = bounds

class Node:
    def __init__(self, kind: str, split_rule: SplitRule = None, children: List[any] = [], classification = ""):
        self.kind = kind # solely for traversal
        self.split_rule = split_rule
        self.children = children
        self.classification = classification
        

def majority_class(classes):
    unique_classes, counts = histogram(classes)
    return unique_classes[idx_of_max(counts)]

def entropy(y):
    if len(y) == 0: return -inf
    x, totals = histogram(y)
    total = sum(totals)
    f  = lambda acc, a: acc + (a / total) * log2(a / total)
    return -reduce(f, totals, 0.0)

def information_gain(parent, children):
    f = lambda acc, child: acc + (len(child) / len(parent)) * entropy(child)
    weighted_average = reduce(f, children, 0)
    return entropy(parent) - weighted_average


def flatten(arr_of_arr):
    r = []
    for arr in arr_of_arr:
        for n in arr:
            r.append(n)

    return r

def permutations(depth, start_idx, arr):
    if depth == 0: return [[]]
    if depth == 1: return [[n] for n in arr[start_idx:]]
    array_of_arrays = [
        [[arr[start_idx + i]] + thing for thing in permutations(depth - 1, start_idx + i + 1, arr)] 
        for i in range(len(arr) - depth)
    ]
    t = flatten(array_of_arrays)
    return t


# important functiuon
# returns every valid bounds-list that paritions the 
# feature values into 
# -- stipulate that features MUST split into all bounds generated
# just check if onformaiton gain is zero (if best split gains zero info then terminate)

# zero chidlren is impossible, one child gains zero information
# thus only should check two or more children
# first elemnet should always parition into first and last into last
# we test for equality on lower bound so we should ignore first element!
# and replace it with neg inf
# for [1, 2] split in two, bounds should be [ [(-inf, 2), (2, inf)] ]
def make_bounds(feature, number_of_children):
    unique_features, _ = histogram(feature)
    if len(unique_features) < number_of_children: return []
    perms = permutations(number_of_children - 1, 0, unique_features[1:])
    add_lower = lambda perm: [-inf] + perm
    nums = fmap(add_lower, perms)
    bounds = fmap(lambda n: list(zip(n, n[1:] + [inf])), nums)
    return bounds


def split(features, classes, feature_idx, bounds):

    # partition rows of dataset by split boundaries
    def reducer(partitions, features_and_class):
        for idx in range(len(partitions)): 
            if in_bound(bounds[idx], fst(features_and_class)[feature_idx]): 
                partitions[idx][0].append(features_and_class[0]) 
                partitions[idx][1].append(features_and_class[1])
        return partitions

    paritions = reduce(reducer, zip(features, classes), [[[], []] for _ in range(len(bounds))])

    # convert to list of (feature, class)
    return fmap(lambda item: (item[0], item[1]), paritions)

# returns the bounds and ig of best split for features.T[feature_idx]
def best_split(features, classes, feature_idx, max_children):

    # get all the valid bounds to split this feature
    possible_bounds = flatten(list(filter(lambda l: l != [], [ 
        make_bounds(get_col(features, feature_idx), number_of_children) 
        for number_of_children in range(2, max_children + 1)
    ])))

    if len(possible_bounds) == 0: return [], -inf

    splits = [
        split(features, classes, feature_idx, bounds) 
        for bounds in possible_bounds
    ]

    # determine the information gained from each split
    split_information_gains = [information_gain(classes, fmap(snd, s)) for s in splits]

    # determine the idx of the highest information gain 
    best_split_idx = idx_of_max(split_information_gains)

    return possible_bounds[best_split_idx], split_information_gains[best_split_idx]


def create_node(features, classes, max_children, min_sample_split):

    # one unique class remaining, return leaf
    unique_classes, _ = histogram(classes)
    if len(unique_classes) == 1 or len(classes) < min_sample_split: 
        return Node("Leaf", classification = majority_class(classes))

    # get best split of each feature, (number_of_children, information_gain)
    best_splits = [
        best_split(features, classes, idx, max_children) 
        for idx in range(len(features[0]))
    ]

    # determine idx of split feature with maximum information gain
    best_feature_idx = idx_of_max(list(map(snd, best_splits)))
    best_bounds, best_information_gain = best_splits[best_feature_idx]

    # solves infinite recursion error
    if best_information_gain <= 0: return Node("Leaf", classification = majority_class(classes))

    #split on this node
    partitions = split(features, classes, best_feature_idx, best_bounds)

    # create bounds, spilt rule and child nodes (recursive step)
    split_rule = SplitRule(best_feature_idx, best_bounds)
    children = list(map(lambda partition: create_node(partition[0], partition[1], max_children, min_sample_split), partitions))

    return Node("Branch", split_rule, children)

def traverse_node(node: Node, indent = 0):
    print("\t" * (indent), end = "")
    if node.kind == "Leaf": 
        print("Class: " + node.classification)
    else:
        print("RULE >> feature: ", end = "")
        print(str(node.split_rule.feature_idx + 1), end = "")
        print(", bounds: " + str(node.split_rule.bounds))
        for child in node.children: traverse_node(child, indent + 1)

def dict_to_split_rule(sr):
    f = lambda bound: (bound[0], bound[1])
    return SplitRule(
        sr["feature_idx"], 
        list(map(f, sr["bounds"]))
    )

def dict_to_model(d):
    if d["kind"] == "Leaf": return Node("Leaf", classification = d["classification"])
    return Node(
        "Branch", 
        dict_to_split_rule(d["split_rule"]), 
        list(map(dict_to_model, d["children"]))
    )


def import_from_json(filename):
    with open(filename) as file:
        return dict_to_model(json.load(file))

def classify(root: Node, attributes):
    if root.kind == "Leaf": return root.classification
    def get_idx(bounds): 
        if in_bound(bounds[0], attributes[root.split_rule.feature_idx]): return 0
        return 1 + get_idx(bounds[1:]) 
    return classify(root.children[get_idx(root.split_rule.bounds)], attributes)

# returns average accurac when cross validation is performed what?
def cross_validation(filename, fold_count):
    features, classes = load_data(filename)
    accuracies, confusions,  durations = [], [], []
    for i in range(fold_count): 
        print(f"Performing fold: {i + 1} of {fold_count}")
        partition = lambda acc, t: (acc[0] + [t[1]], acc[1]) if t[0] % fold_count == i else (acc[0], acc[1] + [t[1]])
        test_features, train_features = reduce(partition, zip(range(len(features)), features), ([], []))
        test_classes, train_classes = reduce(partition, zip(range(len(classes)), classes), ([], []))

        start = time.time()
        model = create_node(train_features, train_classes, 3)
        durations.append(round(time.time() - start, 2))

        predicted_y = [classify(model, x) for x in test_features]

        accuracies.append(accuracy(predicted_y, test_classes))
        confusions.append(confusion(predicted_y, test_classes))

    return accuracies, confusions,  durations

def main():
    features, classes = load_data("data/train_full.txt")
    test_x, test_y = load_data("data/test.txt")

    for min_features  in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        model = create_node(features, classes, 5, min_features)

        predictions = [classify(model, x) for x in test_x]
        export_to_json(model, f"5-nary-models/{min_features}min.json")
        print(f"Accuracy ({min_features} min feature split)", accuracy(predictions, test_y))


   

if __name__ == '__main__': main()