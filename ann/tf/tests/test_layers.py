# Sebastian Raschka 2016-2017
#
# ann is a supporting package for the book
# "Introduction to Artificial Neural Networks and Deep Learning:
#  A Practical Guide with Applications in Python"
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import unittest
import tensorflow as tf
import numpy as np
from ann.tf import conv2d
from ann.tf import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class TestLayers(unittest.TestCase):

    def test_convlayer_and_fclayer(self):

        SEED = 0

        g = tf.Graph()
        with g.as_default() as g:

            # Placeholders and data formatting
            x = tf.placeholder(tf.float32, shape=[None, 784])
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            y = tf.placeholder(tf.float32, shape=[None, 10])

            # Neural network architecture
            conv1 = conv2d(input_tensor=x_image,
                           output_channels=32,
                           seed=SEED)
            pool1 = tf.nn.max_pool(conv1,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
            cur_dim = 28/2

            conv2 = conv2d(input_tensor=pool1,
                           output_channels=64,
                           seed=SEED)
            pool2 = tf.nn.max_pool(conv2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
            cur_dim /= 2
            cur_dim = int(cur_dim)
            fc_features = cur_dim * cur_dim * 64
            flattened = tf.reshape(pool2, [-1, fc_features])

            fc1 = fully_connected(input_tensor=flattened,
                                  output_nodes=1024,
                                  seed=SEED)
            fc2 = fully_connected(input_tensor=fc1,
                                  output_nodes=10,
                                  seed=SEED)

            # Loss function
            cross_entropy = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits(logits=fc2,
                                                          labels=y))

            # Optimization
            train_step = tf.train.AdamOptimizer(learning_rate=1e-4)\

            # Performance
            correct = tf.equal(tf.argmax(input=fc2, axis=1),
                               tf.argmax(input=y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Run graph
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                batch = mnist.train.next_batch(100, shuffle=False)
                feed_dict = {x: batch[0], y: batch[1]}

                if not i % 100:
                    train_acc = sess.run([accuracy], feed_dict=feed_dict)
                    print('step %d, TrainAcc %.2f' % (i, train_acc[0]))
                    break

                sess.run([train_step], feed_dict=feed_dict)

        assert train_acc[0] == np.array(0.04, dtype=np.float32)


if __name__ == '__main__':
    unittest.main()
