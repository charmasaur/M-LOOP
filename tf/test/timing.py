import tensorflow as tf
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.optimize as opt

# Approximation of GELU.
def gelu_fast(_x):
    return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))

def eval_y(x):
    return abs(x)

# Network architecture
INPUT_DIM = 1
HIDDEN_LAYER_DIMS = [32] * 5
ACTS = [tf.nn.relu] * 5
OUTPUT_DIM = 1

# Training
BATCH_SIZE = 8
TRAIN_SIZE = 8 * 10
TRAIN_REG_CO = 0#.001
TRAINER = tf.train.AdamOptimizer
INITIAL_STD = 0.1
EPOCHS = 100

class Net():
    def __init__(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.coord = tf.train.Coordinator()

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
            self.y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
            self.reg_co = tf.placeholder_with_default(0., shape=[])

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
                  prev_h = act(tf.matmul(prev_h, W) + b)
                return tf.matmul(prev_h, Wout) + bout

            self.y = get_y(self.x)

            # Training.
            def get_loss(y_, y):
                return (
                    tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices=[1]))
                    + self.reg_co * tf.reduce_mean([tf.nn.l2_loss(W) for W in Ws + [Wout]]))
            self.loss_func = get_loss(self.y_, self.y)
            self.train_step = TRAINER().minimize(self.loss_func)

            self.batch_size = BATCH_SIZE
            nbatches = tf.to_int32(tf.ceil(
                tf.to_float(tf.shape(self.x)[0]) / tf.to_float(self.batch_size)))

            def batch(j):
                length = tf.minimum(self.batch_size,
                        tf.shape(self.x)[0] - j * self.batch_size)
                xb = tf.slice(self.x, [j * self.batch_size, 0], [length, -1])
                yb = tf.slice(self.y_, [j * self.batch_size, 0], [length, -1])
                return TRAINER().minimize(get_loss(yb, get_y(xb)))
            
            j = tf.constant(0)
            self.epoch_op = tf.while_loop(
                    lambda j: tf.less(j, nbatches),
                    lambda j: tf.tuple([tf.add(j,1)], control_inputs=[batch(j)])[0],
                    [j],
                    back_prop=False,
                    parallel_iterations=1)

            # Gradient with respect to x.
            self.dydx = tf.gradients(self.y, self.x)[0]

            # Optimising.
            self.best_x = None

            self.init = tf.global_variables_initializer()
            self.init2 = tf.local_variables_initializer()

        self.reset()

    def reset(self):
        self.session.run(self.init)
        self.session.run(self.init2)

    def loss(self, xs, ys, reg=True):
        return self.session.run(self.loss_func, feed_dict={
            self.x: xs,
            self.y_: ys,
            self.reg_co: TRAIN_REG_CO if reg else 0,
            })

    def eval(self, xs):
        return self.session.run(self.y, feed_dict={self.x: xs})

    def eval_grad(self, xs):
        return self.session.run(self.dydx, feed_dict={self.x: xs})

    def train(self, epochs=None, plot=False):
        losses = []

        counter = 0
        tx = np.array(train_x)
        ty = np.array(train_y)
        while counter < epochs:
            all_indices = np.random.permutation(len(train_x))

            #for j in range(math.ceil(len(all_indices) / BATCH_SIZE)):
            #    batch_indices = all_indices[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
            #    batch_x = tx[batch_indices]
            #    batch_y = ty[batch_indices]
            #    self.session.run(self.train_step, feed_dict={
            #        self.x: batch_x,
            #        self.y_: batch_y,
            #        self.reg_co: TRAIN_REG_CO,
            #        })

            state = np.random.get_state()
            np.random.shuffle(tx)
            np.random.set_state(state)
            np.random.shuffle(ty)
            self.session.run(self.epoch_op, feed_dict={
                        self.x: tx,
                        self.y_: ty,
                        self.reg_co: TRAIN_REG_CO,
                        })
            counter += 1

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
        print("Maybe found maximum, f(%f) = %f" % (xopt, -fun(xopt)))
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
    x = np.random.uniform(-0.5,0.5)
    add_x(x + 0.5 if x>=0 else x - 0.5)

def explore_min():
    add_x(best_x[0])

def plot(nets):
    predicted_x = _get_xrange()
    plt.clf()
    for n in nets:
        predicted_y = [r[0] for r in n.eval(predicted_x)]
        plt.plot(predicted_x, predicted_y, color='b')
    plt.scatter(train_x, train_y)
    plt.draw()

# Convenience functions
nets = []
def p():
    plot(nets)
    plt.show()

def a():
    raise ValueError
    optimise()
    explore_min()

def t():
    for n in nets:
        n.train()

def an():
    nets.append(Net())

reset()
for _ in range(TRAIN_SIZE):
    explore_random()
startc = time.time()
nets.append(Net())
print("Constructed: " + str(time.time() - startc))
rstart = time.time()
for _ in range(10):
    start = time.time()
    nets[0].train(EPOCHS)
    print(" Time: " + str(time.time() - start))
print("Time: " + str(time.time() - rstart))

print("Loss: " + str(nets[0].loss(train_x,train_y)))
