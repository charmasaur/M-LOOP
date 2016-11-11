import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np

TRAIN_FN = "train.txt"
TEST_FN = "test.txt"
INPUT_DIM = 1
HIDDEN_LAYER_DIMS = [10, 10, 10, 10]
OUTPUT_DIM = 1
LEARNING_RATE = 0.01
TRAIN_RUNS = 1000

# Load training data.
print("Loading data")
_train = json.load(open(TRAIN_FN, "r"))
train_x = [[t[0]] for t in _train]
train_y = [[t[1]] for t in _train]
_test = json.load(open(TEST_FN, "r"))
test_x = [[t[0]] for t in _test]
test_y = [[t[1]] for t in _test]

# TensorFlow setup.
print("Setting up TF")
# Inputs.
x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])

# Variables.
Ws = []
bs = []

prev_layer_dim = INPUT_DIM
for dim in HIDDEN_LAYER_DIMS:
  Ws.append(tf.Variable(tf.random_normal([prev_layer_dim, dim])))
  bs.append(tf.Variable(tf.random_normal([dim])))
  prev_layer_dim = dim

Wout = tf.Variable(tf.random_normal([prev_layer_dim, OUTPUT_DIM]))
bout = tf.Variable(tf.random_normal([OUTPUT_DIM]))

# Computations.
hs = [x]
for (W, b) in zip(Ws, bs):
  hs.append(tf.nn.sigmoid(tf.matmul(hs[-1], W) + b))
y = tf.matmul(hs[-1], Wout) + bout

# Training.
loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_func)

# Session.
print("Starting session");
session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

def loss(xs, ys):
    return session.run(loss_func, feed_dict={x: xs, y_: ys})

def train(steps=1):
    losses = []
    for i in range(steps):
        session.run(train_step, feed_dict={x: train_x, y_: train_y})
        losses.append(loss(train_x, train_y))
        print("Run %d, loss %f" % (i, losses[-1]))
    plt.plot(losses)
    plt.show()

def plot():
    predicted_x = [[x] for x in np.linspace(2. * min(train_x)[0], 2. * max(train_x)[0], 1000)]
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    plt.scatter(train_x, train_y)
    plt.plot(predicted_x, predicted_y, color='r')
    plt.show()

train(TRAIN_RUNS)
print("Test loss %f" % loss(test_x, test_y))
plot()
