import tensorflow as tf

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:49:27 2017
@author: rohankoodli
"""

import numpy as np
import os
import tensorflow as tf

# tags is list of lists
id = []
tags = []
y = []

print(len(tags))
# Assign ranks to various clothing articles
y[0] = 99
y[1] = 88
y[2] = 93
y[4] = 34
y[5] = 10
y[6] = 51
y[7] = 76
y[8] = 65
y[9] = 43

TRAIN_KEEP_PROB = 1.0
TEST_KEEP_PROB = 0.9
learning_rate = 0.0001
ne = 500

train = 3000
test = 20
num_nodes = 250

n_nodes_hl1 = num_nodes # hidden layer 1
n_nodes_hl2 = num_nodes
n_nodes_hl3 = num_nodes
n_nodes_hl4 = num_nodes
n_nodes_hl5 = num_nodes
n_nodes_hl6 = num_nodes
n_nodes_hl7 = num_nodes
n_nodes_hl8 = num_nodes
n_nodes_hl9 = num_nodes
n_nodes_hl10 = num_nodes

n_classes = 4
batch_size = 100 # load 100 features at a time

TF_SHAPE = 14

x = tf.placeholder('float',[None,TF_SHAPE],name="x_placeholder") # 216 with enc0
y = tf.placeholder('float',name='y_placeholder')
keep_prob = tf.placeholder('float',name='keep_prob_placeholder')

def neuralNet(data):
    hl_1 = {'weights':tf.Variable(tf.random_normal([TF_SHAPE, n_nodes_hl1]),name='Weights1'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl1]),name='Biases1')}

    hl_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]),name='Weights2'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl2]),name='Biases2')}

    hl_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]),name='Weights3'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl3]),name='Biases3')}

    hl_4 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4]),name='Weights4'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl4]),name='Biases4')}

    hl_5 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5]),name='Weights5'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl5]),name='Biases5')}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5, n_classes]),name='Weights-outputlayer'),
            'biases':tf.Variable(tf.random_normal([n_classes]),name='Biases-outputlayer')}

    l1 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    l1 = tf.nn.sigmoid(l1, name='op1')

    l2 = tf.add(tf.matmul(l1, hl_2['weights']), hl_2['biases'])
    l2 = tf.nn.sigmoid(l2, name='op2')

    l3 = tf.add(tf.matmul(l2, hl_3['weights']), hl_3['biases'])
    l3 = tf.nn.sigmoid(l3, name='op3')

    l4 = tf.add(tf.matmul(l3, hl_4['weights']), hl_4['biases'])
    l4 = tf.nn.sigmoid(l4, name='op4')

    l5 = tf.add(tf.matmul(l4, hl_5['weights']), hl_5['biases'])
    l5 = tf.nn.sigmoid(l5, name='op5')

    dropout = tf.nn.dropout(l5,keep_prob, name='op6')
    ol = tf.add(tf.matmul(dropout, output_layer['weights']), output_layer['biases'], name='op7')

    tf.summary.histogram('weights-hl_1',hl_1['weights'])
    tf.summary.histogram('biases-hl_1',hl_1['biases'])
    tf.summary.histogram('act-hl_1',l1)

    tf.summary.histogram('weights-hl_2',hl_2['weights'])
    tf.summary.histogram('biases-hl_2',hl_2['biases'])
    tf.summary.histogram('act-hl_2',l2)

    tf.summary.histogram('weights-hl_3',hl_3['weights'])
    tf.summary.histogram('biases-hl_3',hl_3['biases'])
    tf.summary.histogram('act-hl_3',l3)

    tf.summary.histogram('weights-hl_4',hl_4['weights'])
    tf.summary.histogram('biases-hl_4',hl_4['biases'])
    tf.summary.histogram('act-hl_4',l4)

    tf.summary.histogram('weights-hl_5',hl_5['weights'])
    tf.summary.histogram('biases-hl_5',hl_5['biases'])
    tf.summary.histogram('act-hl_5',l5)

    return ol

print "Training"
def train(x):
    prediction = neuralNet(x)
    #print prediction
    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
        tf.summary.scalar('cross_entropy',cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # learning rate = 0.001

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        tf.summary.scalar('accuracy',accuracy)

    # cycles of feed forward and backprop
    num_epochs = ne

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(int(X.shape[0])/batch_size):#mnist.train.num_examples/batch_size)): # X.shape[0]
                randidx = np.random.choice(X.shape[0], batch_size, replace=False)
                epoch_x,epoch_y = X[randidx,:],y[randidx,:] #mnist.train.next_batch(batch_size) # X,y
                j,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                if i == 0:
                    [ta] = sess.run([accuracy],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                    print 'Train Accuracy', ta

                epoch_loss += c
            print '\n','Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss

        saver.save(sess, os.getcwd()+'/tfmodel')
        saver.export_meta_graph(os.getcwd()+'/models/tfmodel.meta')

        print ('\n','Train Accuracy', accuracy.eval(feed_dict={x:tags, y:y, keep_prob:TRAIN_KEEP_PROB}))
        print ('\n','Test Accuracy', accuracy.eval(feed_dict={x:test_X, y:test_y, keep_prob:1.0}))
