import tensorflow as tf
import json
import math
import matplotlib.pyplot as plt
import numpy as np

TRAIN_FN = "train.txt"
TEST_FN = "test.txt"
INPUT_DIM = 1
HIDDEN_LAYER_DIMS = [10, 10, 10, 10]
OUTPUT_DIM = 1
BATCH_SIZE = 100
TRAIN_EPOCHS = 300
OPTIMISE_EPOCHS = 1000
TRAINER = tf.train.AdamOptimizer()
OPTIMISER = tf.train.AdamOptimizer()

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

def train(epochs=1):
    losses = []
    for i in range(epochs):
        all_indices = np.random.permutation(len(train_x))
        for j in range(math.ceil(len(all_indices) / BATCH_SIZE)):
            batch_indices = all_indices[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
            batch_x = [train_x[index] for index in batch_indices]
            batch_y = [train_y[index] for index in batch_indices]
            session.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        losses.append(math.log(loss(train_x, train_y)))
        print("Training epoch %d, loss %f" % (i, losses[-1]))
    plt.plot(losses)
    plt.show()

def optimise(epochs=1):
    x_vals = []
    y_vals = []
    for i in range(epochs):
        session.run(opt_step)
        x_vals.append(session.run(x_opt)[0])
        y_vals.append(session.run(y_opt)[0])
        print("Optimising run %d, f(%f) = %f" % (i, x_vals[-1], y_vals[-1]))

def plot():
    predicted_x = [[x] for x in np.linspace(1.5 * min(train_x)[0], 1.5 * max(train_x)[0], 1000)]
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    plt.scatter(train_x, train_y)
    plt.plot(predicted_x, predicted_y, color='r')
    plt.scatter(session.run(x_opt)[0], session.run(y_opt)[0], color='g')
    plt.show()

def plotgrad():
    predicted_x = [[x] for x in np.linspace(1.5 * min(train_x)[0], 1.5 * max(train_x)[0], 1000)]
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    predicted_grad = [r[0] for r in session.run(dydx, feed_dict={x: predicted_x})]
    plt.plot(predicted_x, predicted_y, color='r')
    plt.plot(predicted_x, predicted_grad, color='b')
    plt.show()


train(TRAIN_EPOCHS)
print("Test loss %f" % loss(test_x, test_y))
optimise(OPTIMISE_EPOCHS)
plot()
