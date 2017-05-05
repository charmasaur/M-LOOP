import tensorflow as tf
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

# Approximation of GELU.
def gelu_fast(_x):
    return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))

# Input files
TRAIN_FN = "train.txt"

# Network architecture
INPUT_DIM = 1
HIDDEN_LAYER_DIMS = [512] * 2
ACTS = [tf.nn.relu] * 2
OUTPUT_DIM = 1

# Training
BATCH_SIZE = 100
TRAIN_EPOCHS_PER_POINT = 1000
TRAIN_KEEP_PROB = 0.9
TRAIN_REG_CO = 0#0.001
TRAINER = tf.train.AdamOptimizer()

# Optimisation
OPTIMISE_EPOCHS = 100
OPTIMISER = tf.train.GradientDescentOptimizer(1.)

LCB_REPS = 100
LCB_PERCENTILE = 0.1

# Load training data.
print("Loading data")
_data = json.load(open(TRAIN_FN, "r"))
data_x = [[t[0]] for t in _data]
data_y = [[t[1]] for t in _data]

# TensorFlow setup.
print("Setting up TF")
# Inputs.
x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
keep_prob = tf.placeholder_with_default(1., shape=[])
reg_co = tf.placeholder_with_default(0., shape=[])
y_ran = tf.placeholder_with_default(1., shape=[])
y_offset = tf.placeholder_with_default(0., shape=[])

# Variables.
Ws = []
bs = []

prev_layer_dim = INPUT_DIM
for dim in HIDDEN_LAYER_DIMS:
  Ws.append(tf.Variable(tf.random_normal([prev_layer_dim, dim], stddev=0.1)))
  bs.append(tf.Variable(tf.random_normal([dim])))
  prev_layer_dim = dim

Wout = tf.Variable(tf.random_normal([prev_layer_dim, OUTPUT_DIM]))
bout = tf.Variable(tf.random_normal([OUTPUT_DIM]))

# Computations.

# Use a function to generate a y variable as a function of an x variable so that we can generate
# multiple variable pairs (one for training, one for optimising, etc...).
def get_y(x_var):
  prev_h = x_var
  for (W, b, act) in zip(Ws, bs, ACTS):
    prev_h = tf.nn.dropout(act(tf.matmul(prev_h, W) + b), keep_prob=keep_prob)
  return tf.matmul(prev_h, Wout) + bout

y = get_y(x)

# Training.
loss_func = (
        (tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
        + reg_co * tf.reduce_mean([tf.nn.l2_loss(W) for W in Ws + [Wout]]))
        * (1 + (y_offset - tf.reduce_max(y_)) * y_ran))
train_step = TRAINER.minimize(loss_func)

# Gradient with respect to x.
dydx = tf.gradients(y, x)[0]

# Find x to maximise the predicted value.
x_opt = tf.Variable(tf.random_normal([1, INPUT_DIM]))
y_opt = get_y(x_opt)
x_opt_manual = tf.placeholder(tf.float32, shape=[1, INPUT_DIM])
set_x_opt = x_opt.assign(x_opt_manual)
reset_opt_step = x_opt.assign(tf.random_uniform([1, INPUT_DIM], -10, 10))
opt_step = OPTIMISER.minimize(-y_opt, var_list=[x_opt])

# Session.
print("Starting session");
session = tf.InteractiveSession()

def reset():
    global next_data_index, train_x, train_y
    next_data_index = 0
    train_x = []
    train_y = []
    session.run(tf.initialize_all_variables())

def get_ran():
    return 0
    diff = max([t[0] for t in train_y]) - min([t[0] for t in train_y])
    if False and diff > 0:
        return 1/diff
    return 0

def loss(xs, ys, reg=True):
    return session.run(loss_func, feed_dict={x: xs, y_: ys, reg_co: TRAIN_REG_CO if reg else 0,
        y_ran: get_ran(),
        y_offset: max([t[0] for t in train_y])
        })

def train_once():
    all_indices = np.random.permutation(len(train_x))
    for j in range(math.ceil(len(all_indices) / BATCH_SIZE)):
        batch_indices = all_indices[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
        batch_x = [train_x[index] for index in batch_indices]
        batch_y = [train_y[index] for index in batch_indices]
        session.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: TRAIN_KEEP_PROB,
            reg_co: TRAIN_REG_CO,
            y_ran: get_ran(),
            y_offset: max([t[0] for t in train_y])
            })
    this_loss = loss(train_x, train_y)
    print("Log loss %f (unreg %f)" % (math.log(1+this_loss), math.log(1+loss(train_x, train_y, reg=False))))
    return this_loss

def train(epochs=1, plot=False):
    losses = []
    best_loss = -1
    #for i in range(epochs):
    try:
        while True:
            this_loss = train_once()
            best_loss = min(this_loss, best_loss) if best_loss >= 0 else this_loss
            losses.append(math.log(1+this_loss))
    except KeyboardInterrupt:
        pass

    #print("Epochs done, training until log loss is better than %f" % (math.log(1+best_loss)))
    # Keep training until the loss gets better
    #while True:
    #    this_loss = train_once()
    #    losses.append(math.log(1+this_loss))
    #    if this_loss < best_loss:
    #        break
    if plot:
        plt.plot(losses)
        plt.show()

def optimise(epochs=1, plot=False):
    fun = lambda test_x: -session.run(y, feed_dict={x: [test_x]})[0]
    grad = lambda test_x: -session.run(dydx, feed_dict={x: [test_x]})[0]
    #xopt = opt.fmin(fun, train_x[np.argmax(train_y)])
    #xopt = opt.minimize(
    #        fun=fun,
    #        x0=[0],
    #        jac=grad).x
    xopt = opt.basinhopping(
            func=fun,
            x0=[0]).x
    print("Maybe found maximum, f(%f) = %f" % (xopt, -fun(xopt)))
    session.run(set_x_opt, feed_dict={x_opt_manual: [xopt]})
    #losses = []
    #x_vals = []
    #y_vals = []
    #session.run(reset_opt_step)
    #for i in range(epochs):
    #    session.run(opt_step)
    #    x_vals.append(session.run(x_opt)[0])
    #    y_vals.append(session.run(y_opt)[0])
    #    losses.append(session.run(y_opt)[0])
    #    print("Optimising run %d, f(%f) = %f" % (i, x_vals[-1], y_vals[-1]))
    #if plot:
    #    plt.plot(losses)
    #    plt.show()

def _get_xrange():
    _mid_train_x = (max(train_x)[0] + min(train_x)[0]) / 2
    _wid_train_x = max(train_x)[0] - min(train_x)[0]
    min_plot_x = _mid_train_x - _wid_train_x * 0.75
    max_plot_x = _mid_train_x + _wid_train_x * 0.75
    return [[x] for x in np.linspace(min_plot_x, max_plot_x, 1000)]

def _get_ys_dist(xs):
    repeated_xs = xs * LCB_REPS
    print("Getting distribution results")
    repeated_ys = session.run(y, feed_dict={x: repeated_xs, keep_prob: TRAIN_KEEP_PROB})
    print("Got distribution results")
    ys = [repeated_ys[i::len(xs)] for i in range(len(xs))]
    ys_actual = [[a[0] for a in y] for y in ys]
    print("Unzipped")
    for a in ys_actual:
        a.sort()
    print("Sorted")
    return [[y[i] for y in ys_actual] for i in range(LCB_REPS)]

def _get_ys_lcb(xs):
    repeated_xs = xs * LCB_REPS
    print("Getting distribution results")
    repeated_ys = session.run(y, feed_dict={x: repeated_xs, keep_prob: TRAIN_KEEP_PROB})
    print("Got distribution results")
    ys = [repeated_ys[i::len(xs)] for i in range(len(xs))]
    ys_actual = [[a[0] for a in y] for y in ys]
    print("Unzipped")
    res = []
    for a in ys_actual:
        res.append(np.mean(a) - np.std(a))
    print("Sorted")
    return res

def plot():
    predicted_x = _get_xrange()
    #ydist = _get_ys_dist(predicted_x)
    ylcb = _get_ys_lcb(predicted_x)
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    plt.clf()
    #for (i,ys) in enumerate(ydist):
    #    plt.scatter(predicted_x, ys, c=[i]*len(ys), vmin=0, vmax=len(ydist), cmap=plt.get_cmap("viridis"))
    plt.plot(predicted_x, predicted_y, color='r')
    plt.plot(predicted_x, ylcb, color='b')
    plt.scatter(train_x, train_y)
    plt.scatter(session.run(x_opt)[0], session.run(y_opt)[0], color='r', marker='x')
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
def run_online():
    plt.ion()
    for i in range(len(data_x)):
        next_point()
        train(TRAIN_EPOCHS_PER_POINT)
        optimise(OPTIMISE_EPOCHS)
        plot()
        plt.pause(0.05)
    plt.pause(1000)

def run_batch():
    for i in range(len(data_x)):
        next_point()
    train(TRAIN_EPOCHS_PER_POINT, True)
    #optimise(OPTIMISE_EPOCHS, plot=True)
    plot()
    plt.show()

reset()
#run_online()
run_batch()
