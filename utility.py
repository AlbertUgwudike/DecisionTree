import json
from functools import reduce

def snd(t): return t[1]
def fst(t): return t[0]

def idx_of(collection, item):
    counter = 0
    while collection[counter] != item: counter += 1
    return counter

def idx_of_max(collection):
    max_val = max(collection)
    return idx_of(collection, max_val)

def convert(c):
    return float(c) if c.isnumeric() else ord(c)

def in_bound(bound, val): 
    return val >= fst(bound) and val < snd(bound)

def load_data(filename):
    f = open(filename, "r")
    raw = fmap(lambda l: l.strip().split(","), f.readlines())
    f.close()
    features = fmap(lambda r: r[:-1], raw)
    return fmap(lambda r: fmap(int, r), features), fmap(lambda r: r[-1], raw)

class int_encoder(json.JSONEncoder):
    def default(self, obj):
        return int(obj)

def model_to_dict(node):
    return {
        "kind": node.kind,
        "classification": node.classification,
        "split_rule": None if node.split_rule is None else node.split_rule.__dict__,
        "children": list(map(model_to_dict, node.children))
    }

def export_to_json(model, out_name):
    with open(out_name, "w") as file:
        json.dump(model_to_dict(model), file, cls = int_encoder, indent = 4)

def groupBy(f, l):
    if len(l) == 0: return []
    return reduce(lambda acc, a: (acc[:-1] + [acc[-1] + [a]]) if f(acc[-1][-1], a) else (acc + [[a]]), l[1:], [[l[0]]])

# quicksort
def sortBy(f, l):
    if len(l) == 0: return []
    if len(l) == 1: return l[::]
    if len(l) == 2: return [l[0], l[1]] if f(l[0], l[1]) else [l[1], l[0]]

    pivot = l[-1]
    g = lambda acc, a: (acc[0] + [a], acc[1]) if f(a, pivot) else (acc[0], acc[1] + [a])
    lt, rt = reduce(g, l[:-1], ([], []))

    return sortBy(f, lt) + [pivot] + sortBy(f, rt)

def size(iter):
    return len(list(iter))

def fmap(f, iter):
    return type(iter)(map(f, iter))

def get_col(mat, i):
    return fmap(lambda r: r[i], mat)



    
    
    
