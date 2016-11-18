import tensorflow as tf
import json
import math
import matplotlib.pyplot as plt
import numpy as np

# Input files
TRAIN_FN = "train.txt"

# Network architecture
INPUT_DIM = 1
HIDDEN_LAYER_DIMS = [256, 256]
OUTPUT_DIM = 1

# Training
BATCH_SIZE = 100
TRAIN_EPOCHS_PER_POINT = 10
TRAIN_KEEP_PROB = 1
TRAINER = tf.train.AdamOptimizer()

# Optimisation
OPTIMISE_EPOCHS = 100
OPTIMISER = tf.train.GradientDescentOptimizer(1.)

# Load training data.
print("Loading data")
_data = json.load(open(TRAIN_FN, "r"))
next_data_index = 0
data_x = [[t[0]] for t in _data]
data_y = [[t[1]] for t in _data]
train_x = []
train_y = []

# TensorFlow setup.
print("Setting up TF")
# Inputs.
x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
keep_prob = tf.placeholder_with_default(1., shape=[])

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
    prev_h = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(prev_h, W) + b), keep_prob=keep_prob)
  return tf.matmul(prev_h, Wout) + bout

y = get_y(x)

# Training.
loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
train_step = TRAINER.minimize(loss_func)

# Gradient with respect to x.
dydx = tf.gradients(y, x)[0]

# Find x to maximise the predicted value.
x_opt = tf.Variable(tf.random_normal([1, INPUT_DIM]))
y_opt = get_y(x_opt)
opt_step = OPTIMISER.minimize(-y_opt, var_list=[x_opt])

# Session.
print("Starting session");
session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

def loss(xs, ys):
    return session.run(loss_func, feed_dict={x: xs, y_: ys})

def train(epochs=1, plot=False):
    losses = []
    for i in range(epochs):
        all_indices = np.random.permutation(len(train_x))
        for j in range(math.ceil(len(all_indices) / BATCH_SIZE)):
            batch_indices = all_indices[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
            batch_x = [train_x[index] for index in batch_indices]
            batch_y = [train_y[index] for index in batch_indices]
            session.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: TRAIN_KEEP_PROB})
        losses.append(math.log(loss(train_x, train_y)))
        print("Training epoch %d, loss %f" % (i, losses[-1]))
    if plot:
        plt.plot(losses)
        plt.show()

def optimise(epochs=1, plot=False):
    losses = []
    x_vals = []
    y_vals = []
    for i in range(epochs):
        session.run(opt_step)
        x_vals.append(session.run(x_opt)[0])
        y_vals.append(session.run(y_opt)[0])
        losses.append(session.run(y_opt)[0])
        print("Optimising run %d, f(%f) = %f" % (i, x_vals[-1], y_vals[-1]))
    if plot:
        plt.plot(losses)
        plt.show()

def _get_xrange():
    _mid_train_x = (max(train_x)[0] + min(train_x)[0]) / 2
    _wid_train_x = max(train_x)[0] - min(train_x)[0]
    min_plot_x = _mid_train_x - _wid_train_x * 0.75
    max_plot_x = _mid_train_x + _wid_train_x * 0.75
    return [[x] for x in np.linspace(min_plot_x, max_plot_x, 1000)]

def plot():
    predicted_x = _get_xrange()
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    plt.clf()
    plt.scatter(train_x, train_y)
    plt.plot(predicted_x, predicted_y, color='r')
    #plt.scatter(session.run(x_opt)[0], session.run(y_opt)[0], color='r')
    plt.draw()

def plotgrad():
    predicted_x = _get_xrange()
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    predicted_grad = [r[0] for r in session.run(dydx, feed_dict={x: predicted_x})]
    plt.plot(predicted_x, predicted_y, color='r')
    plt.plot(predicted_x, predicted_grad, color='b')
    plt.show()

# Add a point to the training set.
def add_train(nx, ny):
    train_x.append([nx])
    train_y.append([ny])

# Get the next data point and add it to the training set.
def next_point():
    global next_data_index
    add_train(data_x[next_data_index][0], data_y[next_data_index][0])
    next_data_index = next_data_index + 1

# Add training examples one at a time, training after each one.
plt.ion()
for i in range(len(data_x)):
    next_point()
    train(TRAIN_EPOCHS_PER_POINT)
    #optimise(OPTIMISE_EPOCHS)
    plot()
    plt.pause(0.05)
