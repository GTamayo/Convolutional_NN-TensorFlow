import functools
import tensorflow as tf
import src.model.helpers_model as hmodel


# change to other file
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
    def __init__(self, image, label, keep_prob):
        self.image = image
        self.label = label
        self.keep_prob = keep_prob
        # improve that
        self.prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        conv1 = hmodel.conv_layer(self.image, shape=[5, 5, 3, 32])
        conv1_pool = hmodel.max_pool_2x2(conv1)
        conv2 = hmodel.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = hmodel.max_pool_2x2(conv2)
        conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])
        full_1 = tf.nn.relu(hmodel.full_layer(conv2_flat, 1024))
        full1_drop = tf.nn.dropout(full_1, keep_prob=self.keep_prob)
        y_conv = hmodel.full_layer(full1_drop, 10)
        return y_conv

    @define_scope
    def optimize(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                               labels=self.label))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def error(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
