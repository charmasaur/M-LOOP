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
OPTIMISE_RATE = 0.5
TRAIN_RUNS = 1000
OPTIMISE_RUNS = 30

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

# Use a function to generate a y variable as a function of an x variable so that we can generate
# multiple variable pairs (one for training, one for optimising, etc...).
def get_y(x_var):
  prev_h = x_var
  for (W, b) in zip(Ws, bs):
    prev_h = tf.nn.sigmoid(tf.matmul(prev_h, W) + b)
  return tf.matmul(prev_h, Wout) + bout

y = get_y(x)

# Training.
loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_func)

# Gradient with respect to x.
dydx = tf.gradients(y, x)[0]

# Find x to maximise the predicted value.
x_opt = tf.Variable(tf.random_normal([1, INPUT_DIM]))
y_opt = get_y(x_opt)
opt_step = tf.train.AdamOptimizer(OPTIMISE_RATE).minimize(-y_opt, var_list=[x_opt])

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
        print("Training run %d, loss %f" % (i, losses[-1]))
    plt.plot(losses)
    plt.show()

def optimise(steps=1):
    x_vals = []
    y_vals = []
    for i in range(steps):
        session.run(opt_step)
        x_vals.append(session.run(x_opt)[0])
        y_vals.append(session.run(y_opt)[0])
        print("Optimising run %d, f(%f) = %f" % (i, x_vals[-1], y_vals[-1]))

def plot():
    predicted_x = [[x] for x in np.linspace(2. * min(train_x)[0], 2. * max(train_x)[0], 1000)]
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    plt.scatter(train_x, train_y)
    plt.plot(predicted_x, predicted_y, color='r')
    plt.scatter(session.run(x_opt)[0], session.run(y_opt)[0], color='g')
    plt.show()

def plotgrad():
    predicted_x = [[x] for x in np.linspace(2. * min(train_x)[0], 2. * max(train_x)[0], 1000)]
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    predicted_grad = [r[0] for r in session.run(dydx, feed_dict={x: predicted_x})]
    plt.plot(predicted_x, predicted_y, color='r')
    plt.plot(predicted_x, predicted_grad, color='b')
    plt.show()


train(TRAIN_RUNS)
print("Test loss %f" % loss(test_x, test_y))
optimise(OPTIMISE_RUNS)
plot()
