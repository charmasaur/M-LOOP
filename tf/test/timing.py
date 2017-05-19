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
TRAINER = tf.train.AdamOptimizer()
INITIAL_STD = 0.1
EPOCHS = 1000

class Net():
    def __init__(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.coord = tf.train.Coordinator()

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
            self.y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
            self.reg_co = tf.placeholder_with_default(0., shape=[])
            self.epochs = tf.placeholder(tf.int32, shape=())
            self.batch_size = tf.placeholder(tf.int32, shape=())

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
            self.train_step = TRAINER.minimize(self.loss_func)

            nbatches = tf.floordiv(tf.size(self.x), self.batch_size)

            def epoch(i):
                #indices = tf.range(0, tf.shape(self.x)[0])#tf.random_shuffle(tf.range(0, tf.shape(self.x)[0]))
                demx = tf.random_shuffle(self.x, seed=1337)
                demy = tf.random_shuffle(self.y_, seed=1337)
            
                def batch(j):
                    #return TRAINER.minimize(get_loss(tf.gather(self.y_, indices), get_y(tf.gather(self.x, indices))))
                    #b = tf.slice(dem, [0,j * self.batch_size, 0], [2, self.batch_size, 1])
                    #xb, yb = tf.unstack(b, num=2)
                    xb = tf.slice(demx, [j * self.batch_size, 0], [self.batch_size, -1])
                    yb = tf.slice(demy, [j * self.batch_size, 0], [self.batch_size, -1])
                    return TRAINER.minimize(get_loss(yb, get_y(xb)))
                    #X, Y = [tf.reduce_sum(xb), tf.reduce_sum(yb)]
                    #r = tf.scatter_nd_update(res, [[i,j]], [[xb, yb]])
                    #with tf.control_dependencies([r]):
                    #    return tf.no_op()
            
                j = tf.constant(0)
                epoch_op = tf.while_loop(
                        lambda j: tf.less(j, nbatches),
                        lambda j: tf.tuple([tf.add(j,1)], control_inputs=[batch(j)])[0],
                        [j])
                return epoch_op
            
            i = tf.constant(0)
            self.train_many = tf.while_loop(
                    lambda i: tf.less(i, self.epochs),
                    lambda i: tf.tuple([tf.add(i,1)], control_inputs=[epoch(i)])[0],
                    [i])

            #xs_var = tf.Variable([0], dtype=tf.float32, validate_shape=False, trainable=False)
            #ys_var = tf.Variable([0], dtype=tf.float32, validate_shape=False, trainable=False)
            #
            #self.assign = tf.tuple([tf.assign(xs_var, self.x, validate_shape=False),
            #    tf.assign(ys_var, self.y_, validate_shape=False)])
            #
            #R = tf.train.slice_input_producer([xs_var, ys_var], num_epochs=EPOCHS)
            #r = [tf.reshape(r, [1]) for r in R]
            #self.bs = tf.train.batch(r, batch_size=BATCH_SIZE)
            #
            #self.train_many = TRAINER.minimize(get_loss(self.bs[1], get_y(self.bs[0])))

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
        #this_loss = self.loss(train_x, train_y)
        #print("Log loss %f (unreg %f)" % (math.log(1+this_loss), math.log(1+self.loss(train_x, train_y, reg=False))))
        return 0#this_loss

    def eval(self, xs):
        return self.session.run(self.y, feed_dict={self.x: xs})

    def eval_grad(self, xs):
        return self.session.run(self.dydx, feed_dict={self.x: xs})

    def train(self, epochs=None, plot=False):
        losses = []

        #self.session.run(self.assign, feed_dict={self.x: train_x, self.y_: train_y})
        #threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)
        #ct = 0
        #try:
        #    while not self.coord.should_stop():
        #        ct = ct + 1
        #        self.session.run(self.train_many)#, feed_dict={self.reg_co: TRAIN_REG_CO})
        #        #print(self.session.run(self.bs))
        #except tf.errors.OutOfRangeError:
        #    print("Done")
        #finally:
        #    self.coord.request_stop()
        #print(str(ct))
        #self.coord.join(threads)
            
        #self.session.run(self.train_many, feed_dict={
        #            self.x: train_x,
        #            self.y_: train_y,
        #            self.reg_co: TRAIN_REG_CO,
        #            self.batch_size: BATCH_SIZE,
        #            self.epochs: epochs
        #            })

        try:
            counter = 0
            while epochs == None or counter < epochs:
                this_loss = self.train_once()
                losses.append(math.log(1+this_loss))
                counter += 1
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
startc = time.time()
nets.append(Net())
print("Constructed: " + str(time.time() - startc))
start = time.time()
nets[0].train(EPOCHS)
print("Time: " + str(time.time() - start))
print("Loss: " + str(nets[0].loss(train_x,train_y)))
