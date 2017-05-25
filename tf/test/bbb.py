### Bayes by Backprop (https://arxiv.org/pdf/1505.05424.pdf)

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
HIDDEN_LAYER_DIMS = [32] * 4
ACTS = [gelu_fast] * 4
OUTPUT_DIM = 1

# Training
BATCH_SIZE = 3
TRAIN_REG_CO = 0#.001
TRAIN_SAMPLING_CO = 0#.0001
TRAINER = tf.train.AdamOptimizer()
INITIAL_STD = 1

# TensorFlow setup.
print("Setting up TF")
# Inputs.
x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
reg_co = tf.placeholder_with_default(0., shape=[])
sampling_co = tf.placeholder_with_default(0., shape=[])

class DistVar():
    def __init__(self, eps):
        std = 1.4/tf.sqrt(tf.to_float(shape[0]))
        self.mu = tf.Variable(tf.random_normal(eps.get_shape(), stddev=INITIAL_STD))
        self.rho = tf.Variable(tf.ones(eps.get_shape()) * DistVar._rho_from_sigma(std))
        self.eps = eps

    def _rho_from_sigma(sigma):
        return tf.log(tf.exp(sigma) - 1)

    def _sigma(self):
        # TODO: This used to be tf.log(1 + tf.exp(self.rho)). Does this break anything?
        return self.rho

    # Returns the value of this variable under the current sampling.
    def op(self):
        return self.mu + tf.mul(self._sigma(), self.eps)

    # Returns the log probability of the current sampling.
    def lp(self):
        # 2.5 ~= sqrt(2pi)
        return -tf.reduce_sum(tf.log(2.5 * tf.abs(self._sigma())) + tf.square(self.eps))

class EpsManager():
    def __init__(self):
        self.eps = []

    def make_eps(self, shape):
        eps = tf.placeholder_with_default(tf.zeros(shape=shape), shape=shape)
        self.eps.append(eps)
        return eps

    def fill_eps(self, d):
        for e in self.eps:
            d[e] = np.random.normal(size=e.get_shape().as_list())

eps_manager = EpsManager()

def single_var(shape):
    return DistVar(eps_manager.make_eps(shape))

def double_var(shape):
    eps = eps_manager.make_eps(shape)
    return DistVar(eps), DistVar(eps)

# Variables.
Ws = []
bs = []
b2s = []
b3s = []

prev_layer_dim = INPUT_DIM
for dim in HIDDEN_LAYER_DIMS:
  Ws.append(single_var([prev_layer_dim, dim]))
  f, s = double_var([dim])
  bs.append(f)
  b2s.append(s)
  b3s.append(single_var([dim]))
  prev_layer_dim = dim

Wout = single_var([prev_layer_dim, OUTPUT_DIM])
bout = single_var([OUTPUT_DIM])

def fill_eps(d, fill=True):
    if fill:
        eps_manager.fill_eps(d)
    return d

# Computations.

# Use a function to generate a y variable as a function of an x variable so that we can generate
# multiple variable pairs (one for training, one for optimising, etc...).
def get_y(x_var):
  prev_h = x_var
  for (W, b, b2, b3, act) in zip(Ws, bs, b2s, b3s, ACTS):
    prev_h = act(tf.matmul(prev_h, W.op()) + b.op() + b3.op()) + b2.op()
  return tf.matmul(prev_h, Wout.op()) + bout.op()

y = get_y(x)

# Training.
loss_func = (
        # Loss, or -log-likelihood of these weights (given data): -log P(D|w)
        # Obviously we want this to be small, since we want a high likelihood.
        tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))

        # Regularization, or -log of prior for these weights: -log P(w)
        # Also want this to be small, since we want a high prior.
        + reg_co * tf.reduce_mean([tf.nn.l2_loss(W.op()) for W in Ws + [Wout]])

        # Log probability of this sampling: log(q(w|theta))
        # Want this to be small, so that for given values of the above two losses it's worse if this
        # is a likely sampling. Alternatively, this can be seen as a way to encourage exploration.
        + sampling_co * tf.reduce_mean([v.lp() for v in Ws + bs + [Wout] + [bout]])

        # Note that this loss can be negative, because we've dropped the constants
        # associated with the first two terms.
        )
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

def loss(xs, ys, reg=True, sam=True):
    return session.run(loss_func, feed_dict={x: xs,
        y_: ys,
        reg_co: TRAIN_REG_CO if reg else 0,
        sampling_co: TRAIN_SAMPLING_CO if sam else 0,
        })

def train_once():
    all_indices = np.random.permutation(len(train_x))
    for j in range(math.ceil(len(all_indices) / BATCH_SIZE)):
        batch_indices = all_indices[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
        batch_x = [train_x[index] for index in batch_indices]
        batch_y = [train_y[index] for index in batch_indices]
        session.run(train_step, feed_dict=fill_eps({
            x: batch_x,
            y_: batch_y,
            reg_co: TRAIN_REG_CO,
            sampling_co: TRAIN_SAMPLING_CO,
            },
            fill=True))
    this_loss = loss(train_x, train_y)
    #print("Log loss %f (unreg %f)" % (math.log(1+this_loss), math.log(1+loss(train_x, train_y, reg=False))))
    print("Loss %f (unreg %f) (raw %f)" % (
        this_loss,
        loss(train_x, train_y, reg=False),
        loss(train_x, train_y, reg=False, sam=False)))
    return this_loss

def train(epochs=None, plot=False):
    losses = []
    try:
        counter = 0
        while epochs == None or counter < epochs:
            this_loss = train_once()
            losses.append(this_loss)
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
            jac=grad).x
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
    return [[x] for x in np.linspace(min_plot_x, max_plot_x, 1000)]

def plot(num_rand=1):
    predicted_x = _get_xrange()
    predicted_y = [r[0] for r in session.run(y, feed_dict={x: predicted_x})]
    plt.clf()
    plt.plot(predicted_x, predicted_y, color='r')
    for _ in range(num_rand):
        yran = [r[0] for r in session.run(y, feed_dict=fill_eps({x: predicted_x}))]
        plt.plot(predicted_x, yran, color='b')
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
    #x = np.random.uniform(-0.5,0.5)
    #add_x(x + 0.5 if x>=0 else x - 0.5)
    add_x(np.random.uniform(-1,1))

def explore_min():
    add_x(best_x[0])

# Convenience functions
def p(n=1):
    plot(num_rand=n)
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
