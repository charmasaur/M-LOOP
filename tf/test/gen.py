import numpy as np
import json
import math

train_fn = 'train.txt'
test_fn = 'test.txt'
noise_sd = 0.01
train_examples = 10
test_examples = 10

def func(x):
    return x[0] * x[0] + x[1] * x[1];

json.dump([(x, [func(x)]) for x in np.random.uniform(-1, 1, (train_examples, 2)).tolist()], open(train_fn, 'w'))
print("Saved %d train examples" % train_examples)
json.dump([(x, [func(x)]) for x in np.random.uniform([-1, -1], [1, 1], (test_examples, 2)).tolist()], open(test_fn, 'w'))
print("Saved %d test examples" % test_examples)
