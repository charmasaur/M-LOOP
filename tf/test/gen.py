import numpy as np
import json
import math

train_fn = 'train.txt'
test_fn = 'test.txt'
noise_sd = 0.01
train_examples = 1000
test_examples = 10

def func(x):
    return np.sinc(x + 2.) + np.random.normal(0., noise_sd)

json.dump([(x, func(x)) for x in np.random.uniform(-5., 5., train_examples)], open(train_fn, 'w'))
print("Saved %d train examples" % train_examples)
json.dump([(x, func(x)) for x in np.random.uniform(-5., 5., test_examples)], open(test_fn, 'w'))
print("Saved %d test examples" % test_examples)
