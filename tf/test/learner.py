import tensorflow as tf
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

# Approximation of GELU.
def gelu_fast(_x):
    return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))

def eval_y(x):
    return x**2

# Network architecture
INPUT_DIM = 1
HIDDEN_LAYER_DIMS = [32] * 5
ACTS = [tf.nn.relu] * 5
OUTPUT_DIM = 1

# Training
BATCH_SIZE = 16
TRAIN_KEEP_PROB = 1#0.8
TRAIN_REG_CO = 0#.001
TRAINER = tf.train.AdamOptimizer()
INITIAL_STD = 0.1
RAN_SCALE = 0

LCB_REPS = 50

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
  Ws.append(tf.Variable(tf.random_normal([prev_layer_dim, dim], stddev=INITIAL_STD)))
  bs.append(tf.Variable(tf.random_normal([dim], stddev=INITIAL_STD)))
  prev_layer_dim = dim

Wout = tf.Variable(tf.random_normal([prev_layer_dim, OUTPUT_DIM], stddev=INITIAL_STD))
bout = tf.Variable(tf.random_normal([OUTPUT_DIM], stddev=INITIAL_STD))

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
        * (1 + RAN_SCALE * (y_offset - tf.reduce_max(y_)) * y_ran))
train_step = TRAINER.minimize(loss_func)

# Gradient with respect to x.
dydx = tf.gradients(y, x)[0]

# Optimising.
best_x = None

# Session.
print("Starting session");
session = tf.InteractiveSession()

def reset():
    global train_x, train_y
    train_x = []
    train_y = []
    session.run(tf.initialize_all_variables())

def get_ran():
    diff = max([t[0] for t in train_y]) - min([t[0] for t in train_y])
    if diff > 0:
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
        batch_y = np.log([train_y[index] for index in batch_indices])
        session.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: TRAIN_KEEP_PROB,
            reg_co: TRAIN_REG_CO,
            y_ran: get_ran(),
            y_offset: max([t[0] for t in train_y])
            })
    this_loss = loss(train_x, train_y)
    print("Log loss %f (unreg %f)" % (math.log(1+this_loss), math.log(1+loss(train_x, train_y, reg=False))))
    return this_loss

def train(epochs=None, plot=False):
    losses = []
    try:
        counter = 0
        while epochs == None or counter < epochs:
            this_loss = train_once()
            losses.append(math.log(1+this_loss))
            counter += 1
    except KeyboardInterrupt:
        pass

    if plot:
        plt.plot(losses)
        plt.show()

def optimise(plot=False):
    fun = lambda test_x: session.run(y, feed_dict={x: [test_x]})[0]
    grad = lambda test_x: session.run(dydx, feed_dict={x: [test_x]})[0].astype(np.float64)
    #xopt = opt.fmin(fun, train_x[np.argmax(train_y)])
    xopt = opt.minimize(
            fun=fun,
            x0=[np.random.uniform(-1,1)],
            bounds=[(-2,2)],
            jac=grad,
            tol=1e-8).x
    #xopt = opt.basinhopping(
    #        func=fun,
    #        x0=[0]).x
    print("Maybe found maximum, f(%f) = %f" % (xopt, -fun(xopt)))
    global best_x
    best_x = xopt

def _get_xrange():
    _mid_train_x = (max(train_x)[0] + min(train_x)[0]) / 2
    _wid_train_x = max(train_x)[0] - min(train_x)[0]
    min_plot_x = _mid_train_x - _wid_train_x * 0.75
    max_plot_x = _mid_train_x + _wid_train_x * 0.75
    return [[x] for x in np.linspace(min_plot_x, max_plot_x, 10000)]

def _get_ys_dist(xs):
    repeated_xs = xs * LCB_REPS
    print("Getting distribution results")
    repeated_ys = np.exp(session.run(y, feed_dict={x: repeated_xs, keep_prob: TRAIN_KEEP_PROB}))
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
    repeated_ys = np.exp(session.run(y, feed_dict={x: repeated_xs, keep_prob: TRAIN_KEEP_PROB}))
    print("Got distribution results")
    ys = [repeated_ys[i::len(xs)] for i in range(len(xs))]
    ys_actual = [[a[0] for a in y] for y in ys]
    print("Unzipped")
    mean = [np.mean(a) for a in ys_actual]
    lcb = [np.mean(a) - np.std(a) for a in ys_actual]
    return (mean, lcb)

def plot():
    predicted_x = _get_xrange()
    #ydist = _get_ys_dist(predicted_x)
    ylcb = _get_ys_lcb(predicted_x)
    predicted_y = np.exp([r[0] for r in session.run(y, feed_dict={x: predicted_x})])
    plt.clf()
    #for (i,ys) in enumerate(ydist):
    #    plt.scatter(predicted_x, ys, c=[i]*len(ys), vmin=0, vmax=len(ydist), cmap=plt.get_cmap("viridis"))
    plt.plot(predicted_x, predicted_y, color='r')
    plt.plot(predicted_x, ylcb[1], color='b')
    plt.plot(predicted_x, ylcb[0], color='g')
    plt.scatter(train_x, train_y)
    plt.scatter(best_x, eval_y(best_x), color='r', marker='x')
    plt.draw()
    #if OUTPUT_DIM > 1:
    #    print("Can't plot with output dim > 1")
    #    return
    #if INPUT_DIM > 2:
    #    print("Can't plot with input dim > 2")
    #    return
    #if INPUT_DIM == 1:
    #    predicted_x = _get_xrange()
    #    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    #    plt.clf()
    #    plt.scatter(train_x, train_y)
    #    plt.plot(predicted_x, predicted_y, color='r')
    #    plt.scatter(session.run(x_opt)[0], session.run(y_opt)[0], color='r', marker='x')
    #    plt.draw()
    #elif INPUT_DIM == 2:
    #    ext = 2
    #    predicted_x = [(x0,x1) for x0 in np.linspace(-ext,ext,50) for x1 in np.linspace(-ext,ext,50)]
    #    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    #    miny = min(predicted_y)
    #    maxy = max(predicted_y)
    #    plt.clf()
    #    plt.imshow((np.array(predicted_y).reshape(50,50) - miny) / (maxy - miny), extent=(-ext,ext,-ext,ext), cmap='jet')
    #    plt.scatter([x for (x,_) in train_x], [y for (_,y) in train_x])
    #    #plt.scatter(train_x, train_y)
    #    #plt.plot(predicted_x, predicted_y, color='r')
    #    #plt.scatter(session.run(x_opt)[0], session.run(y_opt)[0], color='r', marker='x')
    #    plt.draw()

def plotgrad():
    predicted_x = _get_xrange()
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    predicted_grad = [r[0] for r in session.run(dydx, feed_dict={x: predicted_x})]
    plt.plot(predicted_x, predicted_y, color='r')
    plt.plot(predicted_x, predicted_grad, color='b')
    plt.show()

# Add a point to the training set.
def _add_train(nx, ny):
    train_x.append(nx)
    train_y.append(ny)

def add_x(x):
    _add_train([x], [eval_y(x)])

# Get the next data point and add it to the training set.
def explore_random():
    x = np.random.uniform(-0.5,0.5)
    add_x(x + 0.5 if x>=0 else x - 0.5)

def explore_min():
    add_x(best_x[0])

# Convenience functions
def p():
    plot()
    plt.show()

def a():
    optimise()
    explore_min()

def t():
    train()

reset()
for _ in range(20):
    explore_random()
optimise()
