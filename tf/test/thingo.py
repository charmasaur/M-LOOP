import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np

TRAIN_FN = "train.txt"
TEST_FN = "test.txt"
INPUT_DIM = 1
LAYER_DIM = 100
OUTPUT_DIM = 1
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
W1 = tf.Variable(tf.random_normal([INPUT_DIM, LAYER_DIM]))
b1 = tf.Variable(tf.random_normal([LAYER_DIM]))
W2 = tf.Variable(tf.random_normal([LAYER_DIM, OUTPUT_DIM]))
b2 = tf.Variable(tf.random_normal([OUTPUT_DIM]))

# Computations.
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.matmul(h, W2) + b2

# Training.
loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.5).minimize(loss_func)

# Session.
print("Starting session");
session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

def loss(xs, ys):
    return session.run(loss_func, feed_dict={x: xs, y_: ys})

def train(steps=1):
    for i in range(steps):
        session.run(train_step, feed_dict={x: train_x, y_: train_y})
        print("Run %d, loss %f" % (i, loss(train_x, train_y)))

def plot():
    predicted_x = [[x] for x in np.linspace(2. * min(train_x)[0], 2. * max(train_x)[0], 1000)]
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    plt.scatter(train_x, train_y)
    plt.plot(predicted_x, predicted_y, color='r')
    plt.show()

train(TRAIN_RUNS)
print("Test loss %f" % loss(test_x, test_y))
plot()
