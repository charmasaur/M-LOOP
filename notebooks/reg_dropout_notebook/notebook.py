'''
IPython notebook investigating the spaz subnet effect when we use regularisation (the "reg_dropout"
data set).

Here there seems to be a small spaz subnet effect in the last layer: all subnets fit the data
similarly, so we will get low variance on the data, but a few of them vary quite wildly away
from the data and thus contribute to higher variance there. Note that the difference isn't huge
in relative terms though (even though it looks big away from the data it's largely just due to
the fact that the function is higher there).

To run, open up IPython and run learner.py, then run this, then run ploty, plotcontrib and plotdllo
as you please.
'''
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
saver.restore(session, "./reg_dropout.ckpt")

actis=[x];outs=[x];douts=[x];
for (W, b, act) in zip(Ws, bs, ACTS):
    actis.append(tf.matmul(douts[-1], W) + b)
    outs.append(act(actis[-1]))
    douts.append(tf.nn.dropout(outs[-1], keep_prob=keep_prob))
    
def _get_ysa(xs,yfun):
    rx = xs * LCB_REPS
    ry = session.run(yfun, feed_dict={x: rx, keep_prob: TRAIN_KEEP_PROB})
    ys = [ry[i::len(xs)] for i in range(len(xs))]
    ysa = [[a[0] for a in y] for y in ys]
    return ysa

xs = _get_xrange()
# We can plot the function and confidence interval, as well as the relative std. The latter shows
# that at the edges the relative std increases.
def ploty():
    plt.figure(1)
    ysa = _get_ysa(xs,y)
    plt.plot(xs, [np.mean(a) for a in ysa])
    plt.plot(xs, [np.mean(a) - np.std(a) for a in ysa])
    plt.plot(xs, [np.mean(a) + np.std(a) for a in ysa])
    plt.plot(xs, [np.abs(np.std(a)/np.mean(a)) for a in ysa])
    plt.scatter(train_x,train_y,zorder=100)
    plt.title("y and confidence interval")
    plt.show(block=False)

# Here we plot the individual contributors. We see that they all agree on the data, but away from
# the data (especially off the edges) there are a few nets that vary more.
def plotcontrib():
    plt.figure(2)
    dems = []
    for i in range(32):
        f = tf.transpose([douts[-1][:,i]])
        print("Doing number " + str(i))
        dems.append(_get_ysa(xs,f))
   
    Wout_done = session.run(Wout)[:,0]
    bout_done = session.run(bout)[0]
    for i,d in enumerate(dems):
        print("Doing number " + str(i))
        plt.plot(xs,np.array([np.mean(a) for a in d]) * Wout_done[i] + bout_done)
    plt.show(block=False)
