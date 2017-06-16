import tensorflow as tf
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

# Approximation of GELU.
def gelu_fast(_x):
    return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))

def hat(x):
    return 1 if x > 0 and x < 1 else 0

def eval_y(x):
    return hat(x+4) - hat(x+2) + hat(x) - hat(x-2) + hat(x-4)
    #return np.sin(2.*np.pi*x)
    #return (x)**2 - 0.5

# Network architecture
INPUT_DIM = 1
ACT = tf.nn.relu
OUTPUT_DIM = 1

# Training
BATCH_SIZE = 16
TRAIN_REG_CO = 0.#.0001
TRAINER = tf.train.AdamOptimizer()
INITIAL_STD = 0.5

class Net():
    def __init__(self, hidden_layer_dims):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
            self.y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
            self.reg_co = tf.placeholder_with_default(0., shape=[])

            # Variables.
            Ws = []
            bs = []

            prev_layer_dim = INPUT_DIM
            for dim in hidden_layer_dims:
              Ws.append(tf.Variable(tf.random_normal(
                  [prev_layer_dim, dim], stddev=1.4/np.sqrt(prev_layer_dim))))
              bs.append(tf.Variable(tf.random_normal([dim], stddev=INITIAL_STD)))
              prev_layer_dim = dim

            Wout = tf.Variable(tf.random_normal(
                [prev_layer_dim, OUTPUT_DIM], stddev=1.4/np.sqrt(prev_layer_dim)))
            bout = tf.Variable(tf.random_normal([OUTPUT_DIM], stddev=INITIAL_STD))

            # Computations.

            # Use a function to generate a y variable as a function of an x variable so that we can generate
            # multiple variable pairs (one for training, one for optimising, etc...).
            def get_y(x_var):
                prev_h = x_var
                for (W, b) in zip(Ws, bs):
                  prev_h = ACT(tf.matmul(prev_h, W) + b)
                return tf.matmul(prev_h, Wout) + bout

            self.y = get_y(self.x)

            # Training.
            self.loss_func = (
                    tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_), reduction_indices=[1]))
                    + self.reg_co * tf.reduce_mean([tf.nn.l2_loss(W) for W in Ws + [Wout]]))
            self.train_step = TRAINER.minimize(self.loss_func)

            # Gradient with respect to x.
            self.dydx = tf.gradients(self.y, self.x)[0]

            # Optimising.
            self.best_x = None

            self.init = tf.global_variables_initializer()
        self.reset()

    def destroy(self):
        self.session.close()

    def reset(self):
        self.session.run(self.init)

    def loss(self, xs, ys, reg=True):
        return self.session.run(self.loss_func, feed_dict={
            self.x: xs,
            self.y_: ys,
            self.reg_co: TRAIN_REG_CO if reg else 0,
            })

    def train_once(self):
        all_indices = np.random.permutation(len(train_x))
        for j in range(math.ceil(len(all_indices) / BATCH_SIZE)):
            batch_indices = all_indices[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
            batch_x = [train_x[index] for index in batch_indices]
            batch_y = [train_y[index] for index in batch_indices]
            self.session.run(self.train_step, feed_dict={
                self.x: batch_x,
                self.y_: batch_y,
                self.reg_co: TRAIN_REG_CO,
                })
        this_loss = self.loss(train_x, train_y)
        print("Log loss %f (unreg %f)" % (math.log(1+this_loss), math.log(1+self.loss(train_x, train_y, reg=False))))
        return this_loss

    def eval(self, xs):
        return self.session.run(self.y, feed_dict={self.x: xs})

    def eval_grad(self, xs):
        return self.session.run(self.dydx, feed_dict={self.x: xs})

    def train(self, epochs=None, plot=False):
        losses = []
        try:
            counter = 0
            while epochs == None or counter < epochs:
                this_loss = self.train_once()
                losses.append(math.log(1+this_loss))
                counter += 1
                #print("Loss: " + str(math.log(1+this_loss)))
        except KeyboardInterrupt:
            pass

        if plot:
            plt.plot(losses)
            plt.show()

    def optimise(self, plot=False):
        fun = lambda test_x: self.eval([test_x])[0]
        grad = lambda test_x: self.eval_grad([test_x])[0].astype(np.float64)

        xopt = opt.minimize(
                fun=fun,
                x0=[np.random.uniform(-1,1)],
                bounds=[(-2,2)],
                jac=grad).x
        #xopt = opt.basinhopping(
        #        func=fun,
        #        x0=[0]).x
        print("Maybe found maximum, f(%f) = %f" % (xopt, fun(xopt)))
        global best_x
        best_x = xopt

def reset():
    global train_x, train_y
    train_x = []
    train_y = []

def _get_xrange():
    _mid_train_x = (max(train_x)[0] + min(train_x)[0]) / 2
    _wid_train_x = max(train_x)[0] - min(train_x)[0]
    min_plot_x = _mid_train_x - _wid_train_x * 0.75
    max_plot_x = _mid_train_x + _wid_train_x * 0.75
    return [[x] for x in np.linspace(min_plot_x, max_plot_x, 1000)]

# Add a point to the training set.
def _add_train(nx, ny):
    train_x.append(nx)
    train_y.append(ny)

def add_x(x):
    _add_train([x], [eval_y(x)])

# Get the next data point and add it to the training set.
def explore_random():
    add_x(np.random.uniform(-5,7))
    #x = np.random.uniform(0, 1)
    #add_x(x * 2 if x >= 0.5 else -(1-x)*2)

def p(ns=[]):
    plot(ns)

def explore_min():
    add_x(best_x[0])

def plot(nets):
    predicted_x = _get_xrange()
    plt.clf()
    for n in nets:
        predicted_y = [r[0] for r in n.eval(predicted_x)]
        plt.plot(predicted_x, predicted_y)
    plt.scatter(train_x, train_y, zorder=100)
    #plt.scatter([best_x], [eval_y(best_x)], marker='x', color='r')
    plt.show()

# Convenience functions
def diff(net):
    p = np.float(net.eval([best_x]))
    a = np.float(eval_y(best_x))
    r = np.abs((p-a)/a)
    print("Pred: " + str(p) + ", act: " + str(a) + ", rat: " + str(r))
    return r

reset()
for _ in range(80):
    explore_random()
