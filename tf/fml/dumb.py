'''
This implements variable batch_size and epochs. Maybe not as elegant as the queue one, but
less magic and less thready.
'''
import tensorflow as tf

sess = tf.InteractiveSession()

xs = tf.placeholder(tf.float32, shape=[None]) # 10
ys = tf.placeholder(tf.float32, shape=[None])
epochs = tf.placeholder(tf.int32, shape=()) # 10
batch_size = tf.placeholder(tf.int32, shape=()) # 2

res = tf.Variable(tf.zeros([10, 5, 2, 2]))

def epoch(i):
    # TODO: Doesn't work if dims are different.
    dem = tf.transpose(tf.random_shuffle(tf.transpose([xs, ys])))

    def batch(j):
        b = tf.slice(dem, [0,j * batch_size], [2, batch_size])
        xb, yb = tf.unstack(b, num=2)
        X, Y = [tf.reduce_sum(xb), tf.reduce_sum(yb)]
        r = tf.scatter_nd_update(res, [[i,j]], [[xb, yb]])
        with tf.control_dependencies([r]):
            return tf.no_op()

    j = tf.constant(0)
    epoch_op = tf.while_loop(
            lambda j: tf.less(j, tf.floordiv(tf.size(xs), batch_size)),
            lambda j: tf.tuple([tf.add(j,1)], control_inputs=[batch(j)])[0],
            [j])
    return epoch_op

i = tf.constant(0)
it = tf.while_loop(
        lambda i: tf.less(i, epochs),
        lambda i: tf.tuple([tf.add(i,1)], control_inputs=[epoch(i)])[0],
        [i])

init = tf.global_variables_initializer()
