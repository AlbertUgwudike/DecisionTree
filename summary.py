from typing import List, Tuple
from utility import load_data, idx_of, groupBy, sortBy, size, fmap
from functools import reduce
import numpy as np
import time
from operator import itemgetter

# python would be so much better with types
# summarise() should return a dictionary containing metrics
# entire dataset - number of entries, number of features, class-list, class histogram
# each feature - ordered values taken (includes min max), information?, value hist 

def pretty_print(headers, data, out_stream = None):
    p = lambda l: out_stream.write("\n" + l) if out_stream != None else print(l)
    fstring = "".join("{" + str(n) + ":<10}" for n in range(len(headers)))
    print("")
    p(fstring.format(*headers))
    for x in zip(*data): 
        p(fstring.format(*x))

def print_histogram(xs, ys, out_stream = None):
    percentages = fmap(lambda n: round(100 * n / ys.sum(), 2))
    pretty_print(["class", "count", "percentage"], [xs, ys, percentages], out_stream)

def histogram(data: List[any]) -> Tuple[List[any], List[int]]:
    x = sortBy(lambda a, b: a < b, [*set(data)])
    counts = [size(filter(lambda d: d == c, data)) for c in x]
    return x, counts

def summarise_feature(feature):
    ordered_values = np.unique(feature)
    value_hist = histogram(feature)
    return { "ordered_values": ordered_values, "information": 0, "value_hist": value_hist }
    

def summarise(filename):
    x, y = load_data(filename)
    entry_count, feature_count = x.shape
    ordered_classes = np.unique(y)
    class_hist = histogram(y)
    feature_summaries = [summarise_feature(ft) for ft in x.T]
    return {
        "name": filename, "entry_count": entry_count,
        "ordered_classes": ordered_classes, "class_hist": class_hist,
        "feature_summaries": feature_summaries, "feature_count": feature_count
    }


# each feature - ordered values taken (includes min max), information?, value hist 
def print_summary(summary, out_stream = None):
    p = lambda l: out_stream.write("\n" + l) if out_stream != None else print(l)
    p(f"\nSummary for {summary['name']}...")
    p(f"\nEntry Count: {summary['entry_count']}")
    p(f"Feature Count: {summary['feature_count']}")
    p(f"Ordered Classes: {summary['ordered_classes']}")
    p("Class Histogram")
    print_histogram(*summary['class_hist'], out_stream = out_stream)
    p("\n------ Feature Summaries -------")
    for i, feature_summary in enumerate(summary['feature_summaries']):
        p(f"\n--- Feature { i + 1 }: ---")
        p(f"Ordered Values: {feature_summary['ordered_values']}")
        p("Value Histogram:")
        print_histogram(*feature_summary['value_hist'], out_stream = out_stream)

def confusion(predicted_y, valid_y):
    valid_classes, _ = histogram(valid_y)
    number_valid_classes = len(valid_classes)
    predicted_classes, _ = histogram(predicted_y)
    pred_val_pairs = list(zip(predicted_y, valid_y))

    def count(pred, valid):
        return len(list(filter(lambda tup: tup[0] == pred and tup[1] == valid, pred_val_pairs)))

    integer_matrix = np.zeros((number_valid_classes, number_valid_classes), dtype = 'int')

    for r, c in [(r, c) for r in range(number_valid_classes) for c in range(number_valid_classes)]:
        integer_matrix[r, c] = count(valid_classes[r], valid_classes[c])

    # convert to characters, add header and sider
    character_matrix = np.vectorize(str)(integer_matrix)
    with_header = np.concatenate((valid_classes.reshape((1, len(valid_classes))), character_matrix), 0)
    sider = np.concatenate((np.array([[" "]]), valid_classes.reshape((len(valid_classes), 1))), 0)
    with_sider  = np.concatenate((sider, with_header), 1)
    
    return with_sider

def print_confusion_matrix(confusion_matrix, out = None):
    p = lambda l: out.write("\n" + l) if out != None else print(l)
    for row in confusion_matrix:
        p(("{: >5}" * len(row)).format(*row))
    

def macro_av_accuracy(predicited_y, valid_y):
    f = lambda acc, a: acc + (1 if a[0] == a[1] else 0)
    return 100 * reduce(f, zip(predicited_y, valid_y), 0) / len(valid_y)


def pair_sort_group(predicted_y, valid_y):
    ordering = lambda a, b: a["pred"] < b["pred"]
    grouping = lambda a, b: a["pred"] == b["pred"]
    paired_y = list(map(lambda tup: { "pred": tup[0], "true": tup[1] }, zip(predicted_y, valid_y)))
    sort_paired_y = sortBy(ordering, paired_y)
    return dict(zip(histogram(predicted_y)[0], groupBy(grouping, sort_paired_y)))

def true_positives(category, pairs):
    return size(filter(lambda pair: pair["pred"] == pair["true"] == category, pairs))

def false_positives(category, pairs):
    return size(filter(lambda pair: pair["true"] != pair["pred"] == category, pairs))

def true_negatives(category, pairs):
    return size(filter(lambda pair: pair["true"] == pair["pred"] != category, pairs))

def false_negatives(category, pairs):
    return size(filter(lambda pair: pair["pred"] != pair["true"] != category, pairs))

def get_metric_counts(predicted_y, valid_y):
    valid_classes, _ = histogram(valid_y)
    paired_y = list(map(lambda tup: { "pred": tup[0], "true": tup[1] }, zip(predicted_y, valid_y)))
    funcs = [true_positives, true_negatives, false_positives, false_negatives]
    return dict(zip(
        ["tps", "tns", "fps", "fns"], 
        fmap(lambda f: fmap(lambda c: f(c, paired_y), valid_classes), funcs)
    ))


def accuracy(predicted_y, valid_y):
    valid_classes, _ = histogram(valid_y)
    tps, tns, fps, fns = tuple(get_metric_counts(predicted_y, valid_y).values())
    return valid_classes, [(tp + tn) / (tp + tn + fp + fn) for tp, tn, fp, fn in zip(tps, tns, fps, fns)]

def precision(predicted_y, valid_y):
    valid_classes, _ = histogram(valid_y)
    tps, _, fps, _ = tuple(get_metric_counts(predicted_y, valid_y).values())
    return valid_classes, [tp / (tp + fp) for tp, fp in zip(tps, fps)]

def recall(predicted_y, valid_y):
    valid_classes, _ = histogram(valid_y)
    tps, _, _, fns = tuple(get_metric_counts(predicted_y, valid_y).values())
    return valid_classes, [tp / (tp + fn) for tp, fn in zip(tps, fns)]

def f1(predicted_y, valid_y):
    valid_classes, _ = histogram(valid_y)
    tps, _, fps, fns = tuple(get_metric_counts(predicted_y, valid_y).values())
    return valid_classes, [tp / (tp + (fp + fn) / 2) for tp, fp, fn in zip(tps, fps, fns)]




# def main():
#     print_summary(summarise("./data/train_noisy.txt"))
#     predicted_y, valid_y = ["A", "B", "C", "B"], ["A", "B", "D", "C"]
#     pred_classes, precisions = precision(predicted_y, valid_y)
#     print((pred_classes, precisions))
#     print(macro_av_precision(predicted_y, valid_y))


# if __name__ == '__main__': main()
    
