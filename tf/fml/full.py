'''
Feed in the full data, then can train for a particular number of epochs with a particular batch
size (but these must be fixed integers). [Also haven't implemented epochs yet, but that should be
easy]
'''
import tensorflow as tf

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
coord = tf.train.Coordinator()

xs_p = tf.placeholder(tf.float32, shape=[None])
ys_p = tf.placeholder(tf.float32, shape=[None])

xs_var = tf.Variable([0], dtype=tf.float32, validate_shape=False)
ys_var = tf.Variable([0], dtype=tf.float32, validate_shape=False)

assign = tf.tuple([tf.assign(xs_var, xs_p, validate_shape=False), tf.assign(ys_var, ys_p, validate_shape=False)])

R = tf.train.slice_input_producer([xs_var, ys_var])
r = [tf.reshape(r, [1]) for r in R]
bs = tf.train.batch(r, batch_size=5)

xs = list(range(20))
ys = [x * 10 for x in xs]

def start(xs, ys):
    sess.run(init)
    print("Inited")
    sess.run(assign, feed_dict={xs_p: xs, ys_p: ys})
    print("Assigned")
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

'''
Calling sess.run(bs) in the following cases:

------
xs = list(range(20))
ys = [x * 10 for x in xs]

r = tf.train.slice_input_producer([xs, ys], shuffle=False)
s = tf.train.slice_input_producer([[r[0]], [r[1]]], shuffle=True)
bs = tf.train.batch(s, batch_size=5)

Get the batches in order, because s only ever contains one element.

------
xs = list(range(20))
ys = [x * 10 for x in xs]

r = tf.train.slice_input_producer([xs, ys], shuffle=True)
s = tf.train.slice_input_producer([[r[0]], [r[1]]], shuffle=False)
bs = tf.train.batch(s, batch_size=5)

Get random batches.
'''
