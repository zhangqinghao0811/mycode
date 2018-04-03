# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:56:48 2018

@author: 张庆昊
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 2
LEARNING_RATE_BASE = 0.001#0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 1e-4
TRAINING_STEPS = 2000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "D:/Python/workspace/test/LeNet5model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,                             
        mnist_inference.IMAGE_SIZE,            
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],          
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i%100 == 0:
                with tf.variable_scope('layer6-fc2', reuse=True):
                    fc2_weights = tf.get_variable("weight")
                y_max = tf.argmax(y,1)
                ys_max = tf.argmax(ys,1)
                data_dict = {x: reshaped_xs, y_: ys}
                w_store = sess.run(fc2_weights)
                print('y :',sess.run(y , feed_dict=data_dict))
                print('y :',sess.run(y , feed_dict=data_dict))#Why are the results different?
                print('y_max :', sess.run(y_max, feed_dict=data_dict))
                print('y_max :', sess.run(y_max, feed_dict=data_dict))
                w_store2 = sess.run(fc2_weights)
                print('w is same:', np.array(w_store==w_store2).all())#parameters doesn't change
                correct_prediction = tf.equal(y_max, ys_max)
                print("After %d training step(s), loss on training batch is %f." % (i, loss_value))
                print('CP:',sess.run(correct_prediction,feed_dict=data_dict))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                train_accuracy = sess.run(accuracy,feed_dict=data_dict)
                print("step %d, training accuracy %g"%(i, train_accuracy))
                break
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("dataset/", one_hot=True)
    tf.reset_default_graph()
    train(mnist)


if __name__ == '__main__':
    tf.app.run()