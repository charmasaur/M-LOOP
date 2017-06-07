'''
Notebook for regularised dropout using relu. As usual, run learner, run this,
and then use ploty and plotcontrib.

This notebook also fixes a bug from the other where each subplot used a different
dropout sampling. It sounds like this doesn't matter but actually I think it does
(because different subplots use common previous layers).

Can't quite remember what this notebook shows... Think we get similar shapes to mean
and LCB, which is why dropout is a bit sucky. In the next notebook we'll explore
what happens if we dropout consistently for a whole run, which will enable Thompson
sampling.
'''

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
saver.restore(session, "./reg_dropout_relu.ckpt")

actis=[x];outs=[x];douts=[x];
for (W, b, act) in zip(Ws, bs, ACTS):
    actis.append(tf.matmul(douts[-1], W) + b)
    outs.append(act(actis[-1]))
    douts.append(tf.nn.dropout(outs[-1], keep_prob=keep_prob))

# Returns array of shape (len(yfun), len(xs), LCB_REPS)
def _get_ysa(xs,yfun):
    rx = xs * LCB_REPS
    ry = session.run(yfun, feed_dict={x: rx, keep_prob: TRAIN_KEEP_PROB})
    return ry.reshape((len(xs),LCB_REPS,-1),order='F').transpose([2,0,1]) 

xs = _get_xrange()
Wout_done = session.run(Wout)[:,0]
bout_done = session.run(bout)[0]

dems = _get_ysa(xs,douts[-1])

def ploty():
    res = np.dot(dems.transpose(),Wout_done) + bout_done
    mean = np.mean(res,axis=0)
    std = np.std(res,axis=0)

    plt.figure(1)
    plt.plot(xs,mean)
    plt.plot(xs,mean+std)
    plt.plot(xs,mean-std)
    plt.plot(xs, np.abs(std/mean))
    plt.scatter(train_x,train_y,zorder=100,color='r')
    plt.show(block=False)

def plotcontrib():
    plt.figure(2)
    for d,w in zip(dems,Wout_done):
        plt.plot(xs,np.mean(d,axis=1) * w)
    plt.plot(xs,[bout_done] * len(xs))
    plt.show(block=False)

