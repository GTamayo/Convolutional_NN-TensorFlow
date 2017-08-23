#!/usr/bin/python3
import src.data_load_manager as load_manager
import numpy as np
import src.model.CNN_model as NNmodel
import tensorflow as tf
import argparse


def main(steps, batch_size):
    cifar = load_manager.CifarDataManager()
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    model = NNmodel.Model(x, y_, keep_prob)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            batch = cifar.train.next_batch(batch_size)
            sess.run(model.optimize, feed_dict={x: batch[0], y_: batch[1],
                                                keep_prob: 0.5})
            X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
            Y = cifar.test.labels.reshape(10, 1000, 10)
            acc = np.mean([sess.run(model.error, feed_dict={x: X[i], y_: Y[i], keep_prob: 1}) for i in range(10)])
            print("Accuracy:{:.4}%".format(acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model_train')
    parser.add_argument('-steps',
                        help='number of steps',
                        type=int,
                        default=100)
    parser.add_argument('-batch_size',
                        help='size batch',
                        type=str,
                        default=100)
    args = parser.parse_args()
    main(
        steps=args.steps,
        batch_size=args.batch_size)
