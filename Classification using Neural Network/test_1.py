################ confusion matrix and testing over training #####################
##here we train and test on same dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import math
import time
import data_parser
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os



my_loss=[]
class DataSet(object):
    def __init__(self, X, Y):
        """Construct a DataSet.

        """
        self._num_examples = len(Y)
        self._X = X
        self._Y = Y
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._X = self._X[perm]
            self._Y = self._Y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._Y[start:end]


def placeholder_inputs(FEATURE_NUM):
    X_placeholder = tf.placeholder(tf.float32, shape=(None, FEATURE_NUM))
    Y_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
    return X_placeholder, Y_placeholder

def fill_feed_dict(data_set, X_pl, Y_pl, batch_size):
    X_feed, Y_feed = data_set.next_batch(batch_size)
    feed_dict = {
        X_pl: X_feed,
        Y_pl: Y_feed,
    }
    return feed_dict


def do_eval(sess, eval_loss, X_placeholder, Y_placeholder, data_set, predicted_value, effect, abs_effect):
    Y_pred, loss = sess.run([predicted_value, eval_loss], feed_dict={X_placeholder: data_set.X, Y_placeholder: data_set.Y,})
    return Y_pred, math.sqrt(loss / len(data_set.Y)), effect, abs_effect


def loss_fn(predicted_value, true_value, l2_loss):
    
    return tf.reduce_mean(tf.square(predicted_value - true_value)) + 0.001 * l2_loss
##    return tf.reduce_mean(-tf.reduce_sum(predicted_value * tf.log(true_value), reduction_indices=[1])) + 0.0001 * l2_loss
##    cross_entropy = tf.reduce_mean(-tf.reduce_sum(predicted_value * tf.log(true_value), reduction_indices=[1]))

def evaluation(predicted_value, true_value, needToUnormalize=False):
   
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    # correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    # return tf.reduce_sum(tf.cast(correct, tf.int32))
    if needToUnormalize:
        return tf.reduce_sum(tf.square(predicted_value * 79.7453 + 89.679 - true_value))
    else:
        return tf.reduce_sum(tf.square(predicted_value - true_value))



def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_training(data_train, data_test, learning_rate, batch_size, hidden1, max_steps, FEATURE_NUM, seedNum):

    with tf.Graph().as_default():
        # Randomizing the graph seed number
        tf.set_random_seed(-seedNum)
        # Generate placeholders for the images and labels.
        X_placeholder, Y_placeholder = placeholder_inputs(FEATURE_NUM)
        images, Y_placeholder = placeholder_inputs(FEATURE_NUM)

        hidden1_units = hidden1

        factor = 4
        # Hidden 1
        loss = tf.Variable(tf.zeros([1]))
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.random_uniform([FEATURE_NUM, hidden1_units],
                                  -factor * math.sqrt(6 / (FEATURE_NUM + hidden1_units)),
                                  factor * math.sqrt(6 / (FEATURE_NUM + hidden1_units)), seed=seedNum),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
            hidden1 = tf.nn.sigmoid(tf.matmul(X_placeholder, weights) + biases)
            loss += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
##            loss+=tf.reduce_sum(tf.abs(weights))+tf.reduce_sum(tf.abs(biases))
            

        # Linear
        with tf.name_scope('linear'):
            weights2 = tf.Variable(
                tf.random_uniform([hidden1_units, 1],
                                  -4 * math.sqrt(6 / (hidden1_units + 1)),
                                  4 * math.sqrt(6 / (hidden1_units + 1))),
                name='weights')
            biases = tf.Variable(tf.zeros([1]),
                                 name='biases')
            loss += tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases)
##            loss+=tf.reduce_sum(tf.abs(weights2))+tf.reduce_sum(tf.abs(biases))
            predicted_value = tf.nn.sigmoid(tf.matmul(hidden1, weights2) + biases)
        l2_loss = loss

        # Add to the Graph the Ops for loss calculation.
        
        loss = loss_fn(predicted_value, Y_placeholder, l2_loss)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_loss = evaluation(predicted_value, Y_placeholder)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Creates a saver to save and restore models
        saver = tf.train.Saver()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Training over number of epochs
        for step in range(max_steps):
            feed_dict = fill_feed_dict(data_train, X_placeholder, Y_placeholder, batch_size)

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
##            print(loss_value)
            my_loss.append(loss_value)
            if step % 1000 == 0:
                _, train_one,na,na2 = do_eval(sess, eval_loss, X_placeholder, Y_placeholder, data_train, predicted_value,0,0)
                _, test_one,na,na1 = do_eval(sess, eval_loss, X_placeholder, Y_placeholder, data_test, predicted_value,0,0)


        weights_hidden = np.array(weights.eval(session=sess)) # Weight values to the hidden layer
        weights_output = np.array(weights2.eval(session=sess)).transpose() # Weight values to the output layer
        effect = np.average(weights_hidden*weights_output,axis=1) # (hidden layer weight) * (output layer weight)
        print(effect)
        abs_effect = np.average(np.abs(weights_hidden*weights_output),axis=1) # absolute value of the overall weight
        return do_eval(sess, eval_loss, X_placeholder, Y_placeholder, data_test, predicted_value, effect, abs_effect) # Predicts



featdat,dat,data = data_parser.parse("DBTT_Data19.csv")
X = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(eff fluence))"]
X_LWR = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(eff fluence))"]

Y = "delta sigma"
data.set_x_features(X)
data.set_y_feature(Y)
    
lwr_datapath = "CD_LWR_clean7.csv"
##data.add_exclusive_filter("Alloy", '=', 29)
##data.add_exclusive_filter("Alloy", '=', 14)
##data.add_exclusive_filter("Temp (C)", '<>', 290)
k=[]


trainX = np.asarray(data.get_x_data())
trainY = np.asarray(data.get_y_data())

for i in trainY:
##        k.append([1])
    if(i[0]>=300):
        k.append([1])
    else:
        k.append([0])

new=np.asarray(k)
trainY=new

testX=trainX
testY=trainY
##split=int(0.1*len(trainX))
##testX=trainX[-split:-1]
##trainX=trainX[:-split]
##testY=trainY[-split:-1]
##trainY=trainY[:-split]

add_tr_X=[]
add_tr_Y=[]
for i in range(len(trainX)):
	if(trainY[i][0]==1):
		add_tr_X.append(trainX[i])
		add_tr_Y.append(trainY[i])

add_tr_X=np.array(add_tr_X)
add_tr_Y=np.array(add_tr_Y)
##
oversampling_coeff=20
for i in range(oversampling_coeff):
    for i in range(len(add_tr_X)):
        t=add_tr_X[i].reshape((1,7))
        trainX=np.append(trainX,t,axis=0)
        trainY=np.append(trainY,[[1]],axis=0)
	



train = DataSet(trainX, trainY)
test=DataSet(testX,testY)
##featdat,dat,lwr_data = data_parser.parse(lwr_datapath)
##lwr_data.set_x_features(X)
##lwr_data.set_y_feature(Y)
##lwr_data.add_exclusive_filter("Temp (C)", '<>', 290)
##
##
##lwr_data.add_exclusive_filter("Alloy", '=', 29)
##lwr_data.add_exclusive_filter("Alloy", '=', 14)
##
##testX = np.asarray(lwr_data.get_x_data())
##testY = np.asarray(lwr_data.get_y_data())

hidden1 = [25]
rate = [.5]
epoch = [20000]
batch = [len(trainX)]

## Number of times to predict
z = 5
for a in hidden1:
    for b in rate:
        for c in epoch:
            for d in batch:
                for n in range(1):
                    Ypredict, rms, weights, abs_weights = run_training(train, test, learning_rate = b, batch_size = d, hidden1 = a, max_steps = c, FEATURE_NUM=len(X), seedNum = n)
                    rms_effective = np.sqrt(mean_squared_error(Ypredict, testY))
                    print(rms_effective)
##                    predictions_effective = Ypredict[effective_index]
##                    Yreal_effective = testY[effective_index]
##err=[]
##for i in range(0,1,0.1):
##    pred=[]
##    for no in Ypredict:
##        if (no>i):
##            pred.append([1])
##        else:
##            pred.append([0])
##        effective = np.sqrt(mean_squared_error(pred, testY))
##        err.append(effective)
##
##

                    
##

Ymy=[]

for i in Ypredict:
	if(i[0]>=0.5):
		Ymy.append([1])
	else:
		Ymy.append([0])

pred_1_actual_1=0
pred_1_not_1=0
pred_0_actual_0=0
pred_0_not_0=0



##for i in range(len(Ypredict)):
##    if(Ymy[i][0]==1):
##        if(testY[i][0]==1):
##            pred_1_actual_1+=1
##        else:
##            pred_1_not_1+=1
##    elif(Ymy[i][0]==0):
##        if(testY[i][0]==0):
##            pred_0_actual_0+=1
##        else:
##            pred_0_not_0+=1

##making the confusion matrix
for i in range(len(Ypredict)):
    if(Ymy[i][0]==1):
        if(testY[i][0]==1):
            pred_1_actual_1+=1
        else:
            pred_1_not_1+=1
    elif(Ymy[i][0]==0):
        if(testY[i][0]==0):
            pred_0_actual_0+=1
        else:
            pred_0_not_0+=1
                
            



