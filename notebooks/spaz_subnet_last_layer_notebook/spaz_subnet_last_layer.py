'''
IPython notebook investigating whether we're getting a "spaz subnet" effect in the last layer.
The "spaz subnet" effect is where we have some "good" subnets, which fit the data well and do
whatever they want elsewhere, and some "spaz" subnets, which are zero on the data and do whatever
they want elsewhere. The interesting thing about this is that when combined with dropout, we can
get really low variance on the data but really high variance elsewhere. If there's no spaz subnet
effect (that is, all the subnets contribute to the data) then we must have a baseline variance on
the data due to the dropout giving us an uncertain number of contributing inputs. This baseline
variance is proportional to the square of the function value (vaguely).

In this notebook we see whether this effect is present at the last layer of the "woah" dataset.
We find that there doesn't seem to be a spaz subnet effect in the last layer, but there probably
is the effect somewhere else.

To run, open up IPython and run learner.py, then run this, then run do_regular and do_dllo as you
please.
'''
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
saver.restore(session, "./woah.ckpt")
xs = _get_xrange()

# dllo = Dropout in Last Layer Only
def _get_ydllo_lcb(xs):
    rx = xs * LCB_REPS
    ry = session.run(y_dllo, feed_dict={x: rx, keep_prob: TRAIN_KEEP_PROB})
    ys = [ry[i::len(xs)] for i in range(len(xs))]
    ysa = [[a[0] for a in y] for y in ys]
    mean = [np.mean(a) for a in ysa]
    lcb = [np.mean(a) - np.std(a) for a in ysa]
    return (mean, lcb)

prev_h = x
for (W, b, act) in zip(Ws, bs, ACTS):
    prev_h = act(tf.matmul(prev_h, W) + b)
act_last_layer_no_dropout = prev_h
y_dllo = tf.matmul(tf.nn.dropout(act_last_layer_no_dropout, keep_prob=keep_prob), Wout) + bout

ydllolcb = _get_ydllo_lcb(xs)
ylcb = _get_ys_lcb(xs)

# Plots for regular. The red varies a lot, indicating that there is a spaz subnet effect (that is,
# there's a contribution to variance that can't just be attributed to dropout noise).
def do_regular():
    plt.clf()
    # |1-(u - s)/u|=|s/u|
    plt.plot(xs, np.abs(1-(np.array(ylcb[1])/np.array(ylcb[0]))), color='r')
    plt.scatter(train_x, train_y, zorder=100)
    plt.plot(xs, ylcb[1], color='b')
    plt.plot(xs, ylcb[0], color='g')
    plt.show(block=False)

# Plots for dllo. The red is roughly constant, indicating that there is no (=small) spaz subnet
# effect in the last layer (the variance is largely just due to dropout noise).
def do_dllo():
    plt.clf()
    plt.plot(xs, np.abs(1-(np.array(ydllolcb[1])/np.array(ydllolcb[0]))), color='r')
    plt.scatter(train_x, train_y, zorder=100)
    plt.plot(xs, ydllolcb[1], color='b')
    plt.plot(xs, ydllolcb[0], color='g')
    plt.show(block=False)
