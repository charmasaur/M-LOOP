import numpy as np
import json
import math

train_fn = 'train.txt'
test_fn = 'test.txt'
noise_sd = 0.002
train_examples = 1000
test_examples = 10

def func(x):
#  return math.sin(x) + np.random.normal(0., noise_sd)
#    return 5./(1.+(x-5.)**2) + 10./(1.+x**2) + np.random.normal(0., noise_sd)
    return np.random.normal(0., noise_sd) + (1 if x == 0 else math.sin(x) / x)

#train_f = open(train_fn, 'w')
json.dump([(x, func(x)) for x in np.random.uniform(-15., 15., train_examples)], open(train_fn, 'w'))
print("Saved %d train examples" % train_examples)
json.dump([(x, func(x)) for x in np.random.uniform(-15., 15., test_examples)], open(test_fn, 'w'))
print("Saved %d test examples" % test_examples)
