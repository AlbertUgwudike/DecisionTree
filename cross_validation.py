from utility import load_data
from summary import accuracy, confusion, print_confusion_matrix
from tree import *
from functools import reduce
from time import time
import numpy as np
from scipy import stats

def cross_validation(filename, fold_count):

    _features, _classes = load_data(filename)
    str_features = np.vectorize(str)(_features)
    features_and_classes = np.concatenate((str_features, _classes.reshape(len(_classes), 1)), axis = 1)
    np.random.shuffle(features_and_classes)
    f = np.vectorize(float)
    features, classes = ( f(features_and_classes[:, :-1]), features_and_classes[:, -1]  )

    other_features, other_classes = load_data("data/test.txt")
    all_predictions = []

    accuracies, confusions,  durations = [], [], []
    for i in range(fold_count): 
        print(f"Performing fold: {i + 1} of {fold_count}")
        partition = lambda acc, t: (acc[0] + [t[1]], acc[1]) if t[0] % fold_count == i else (acc[0], acc[1] + [t[1]])
        test_features, train_features = reduce(partition, zip(range(len(features)), features), ([], []))
        test_classes, train_classes = reduce(partition, zip(range(len(classes)), classes), ([], []))
        n = lambda a: np.array(a)
        start = time()
        model = create_node(n(train_features), n(train_classes), 3, 1)
        durations.append(round(time() - start, 2))

        predicted_y = [classify(model, x) for x in n(test_features)]
        other_predictions = [classify(model, x) for x in n(other_features)]

        all_predictions.append(other_predictions)
        accuracies.append(accuracy(predicted_y, test_classes))
        confusions.append(confusion(predicted_y, test_classes))
    
    modal_classes = stats.mode(np.array(all_predictions))[0]
    voting_accuracy = accuracy(modal_classes.reshape((modal_classes.shape[1],)), other_classes)

    return voting_accuracy, accuracies, confusions,  durations

def main():
    # uncomment to run cross validation
    fold_count = 10
    validation_filename = "data/train_full.txt"
    voting_accuracy, accuracies, confusions,  durations = cross_validation(validation_filename, fold_count)
    mean_accuracy = round(np.array(accuracies).mean(), 2)
    stdev = round(np.array(accuracies).std(), 2)

    with open("cross_validation_tuned_new_model.txt", "w") as file:
        file.write(f"{fold_count}-fold Cross validation results")
        file.write(f"\nConducted on on: {validation_filename}")
        for i in range(fold_count):
            file.write(f"\n\nTraining Duration = {durations[i]}s")
            file.write(f"\nAccuracy = {round(accuracies[i], 2)}%")
            file.write("\nConfusion Matrix: ")
            print_confusion_matrix(confusions[i], file)
        file.write(f"\n\nMean subtree accuracy = {mean_accuracy}%")
        file.write(f"\nSubtree accuracy stdev = {stdev}")
        file.write(f"\nVoting based accuracy = {voting_accuracy}%")

if __name__ == "__main__": main()
