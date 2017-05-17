import tensorflow as tf
import time
import json
import math
import matplotlib.pyplot as plt
import numpy as np
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
TRAIN_SIZE = 20

# Training
BATCH_SIZE = 10
TRAIN_REG_CO = 0#.001
TRAINER = tf.train.AdamOptimizer()
INITIAL_STD = 0.1

class Net():
    def __init__(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.tx = tf.placeholder_with_default(tf.zeros([TRAIN_SIZE, INPUT_DIM]), shape=[TRAIN_SIZE,INPUT_DIM])
            self.ty_ = tf.placeholder_with_default(tf.zeros([TRAIN_SIZE, OUTPUT_DIM]), shape=[TRAIN_SIZE,OUTPUT_DIM])
            self.bx = tf.placeholder_with_default(tf.zeros([BATCH_SIZE, INPUT_DIM]), shape=[BATCH_SIZE,INPUT_DIM])
            self.by_ = tf.placeholder_with_default(tf.zeros([BATCH_SIZE, OUTPUT_DIM]), shape=[BATCH_SIZE,OUTPUT_DIM])
            self.x = tf.placeholder_with_default(tf.zeros([1, INPUT_DIM]), shape=[1,INPUT_DIM])
            self.y_ = tf.placeholder_with_default(tf.zeros([1, OUTPUT_DIM]), shape=[1,OUTPUT_DIM])
            self.reg_co = tf.placeholder_with_default(0., shape=[])
            self.epochs = tf.placeholder_with_default(1, shape=[])

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

            def get_loss(x, y):
                return (
                    tf.reduce_mean(tf.reduce_sum(tf.square(get_y(x) - y), reduction_indices=[1]))
                    + self.reg_co * tf.reduce_mean([tf.nn.l2_loss(W) for W in Ws + [Wout]]))

            self.loss_func = get_loss(self.tx, self.ty_)

            # Training.
            def make_training_step():
                print("Hello")
                nexamples = TRAIN_SIZE#tf.shape(self.x)[0]
                #shape = tf.tile([BATCH_SIZE], [tf.floordiv(nexamples, BATCH_SIZE)])
                #remaining = tf.mod(nexamples, BATCH_SIZE)
                #print(shape)
                #print(remaining)
                #print([remaining])
                #shape = tf.cond(
                #        tf.equal(remaining, 0),
                #        lambda: shape,
                #        lambda: tf.concat(0, [shape, [remaining]]))
                xall = tf.split(0, nexamples, self.tx)
                yall = tf.split(0, nexamples, self.ty_)
                xbatches = [tf.concat(0, xall[i*BATCH_SIZE:(i+1)*BATCH_SIZE]) for i in range(int(nexamples/BATCH_SIZE))]
                ybatches = [tf.concat(0, yall[i*BATCH_SIZE:(i+1)*BATCH_SIZE]) for i in range(int(nexamples/BATCH_SIZE))]
                return tf.tuple([tf.zeros([1]), tf.zeros([1])], control_inputs=[TRAINER.minimize(get_loss(x, y)) for (x,y) in zip(xbatches,ybatches)])[0]
            self.train_step = make_training_step()
            #self.many_train_steps = tf.tile(self.train_step, [self.epochs])

            i = tf.constant(0)
            self.many_train_steps = tf.while_loop(
                    lambda i: tf.less(i, self.epochs),
                    lambda i: tf.tuple((tf.add(i,1),make_training_step()))[0],
                    [i])

            self.one_train_step = TRAINER.minimize(get_loss(self.bx, self.by_))

            # Gradient with respect to x.
            self.dydx = tf.gradients(self.y, self.x)[0]

            # Optimising.
            self.best_x = None

            self.init = tf.initialize_all_variables()
        self.reset()

    def reset(self):
        self.session.run(self.init)

    def loss(self, xs, ys, reg=True):
        return self.session.run(self.loss_func, feed_dict={
            self.tx: xs,
            self.ty_: ys,
            self.reg_co: TRAIN_REG_CO if reg else 0,
            })

    def train_once(self):
        self.session.run(self.train_step, feed_dict={
            self.tx: train_x,
            self.ty_: train_y,
            self.reg_co: TRAIN_REG_CO,
            })
        #all_indices = range(len(train_x))#np.random.permutation(len(train_x))
        #for j in range(math.ceil(len(all_indices) / BATCH_SIZE)):
        #    batch_indices = all_indices[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
        #    batch_x = [train_x[index] for index in batch_indices]
        #    batch_y = [train_y[index] for index in batch_indices]
        #    self.session.run(self.one_train_step, feed_dict={
        #        self.bx: batch_x,
        #        self.by_: batch_y,
        #        self.reg_co: TRAIN_REG_CO,
        #        })
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
            print(self.session.run(self.many_train_steps, feed_dict={
                self.tx: train_x,
                self.ty_: train_y,
                self.reg_co: TRAIN_REG_CO,
                self.epochs: epochs,
                }))

            #counter = 0
            #while epochs == None or counter < epochs:
            #    this_loss = self.train_once()
            #    #losses.append(math.log(1+this_loss))
            #    counter += 1
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
nets.append(Net())
start = time.time()
nets[0].train(100)
print("Time: " + str(time.time() - start))
print("Loss: " + str(nets[0].loss(train_x,train_y)))
