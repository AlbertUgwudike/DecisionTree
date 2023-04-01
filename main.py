from tree import create_node, classify, export_to_json, import_from_json
from utility import load_data, fmap
from summary import precision, pretty_print, accuracy, recall, f1

def main():
    x_train, y_train = load_data("data/train_full.txt")
    x_test, y_test = load_data("data/validation.txt")
    # model = create_node(x_train, y_train, 3, 10)
    # export_to_json(model, "models/best_model.json")
    model = import_from_json("models/best_model.json")
    y_pred = [classify(model, x) for x in x_test]

    classes, accuracies = accuracy(y_pred, y_test)
    _, precisions = precision(y_pred, y_test)
    _, recalls = recall(y_pred, y_test)
    _, f1s = f1(y_pred, y_test)

    format_score = lambda s: fmap(lambda n: round(100 * n, 2), s)

    pretty_print(
        ["Class", "Accuracy", "Precision", "Recall", "F1"], 
        [classes, *(fmap(format_score, [accuracies, precisions, recalls, f1s]))]
    )

if __name__ == "__main__": main()


