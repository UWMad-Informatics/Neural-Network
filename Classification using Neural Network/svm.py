
##original with cross validation

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
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")

epoch=range(2000)
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


featdat,dat,data = data_parser.parse("DBTT_Data19.csv")
X = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(eff fluence))"]
X_LWR = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(eff fluence))"]

Y = "delta sigma"
data.set_x_features(X)
data.set_y_feature(Y)
print("yayy")    
    
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

split=int(0.2*len(trainX))
testX=trainX[-split:-1]
trainX=trainX[:-split]
testY=trainY[-split:-1]
trainY=trainY[:-split]
clf=svm.SVC()
clf.fit(trainX,trainY)


                    
#############################DATA ANALYSIS#########################

Ymy=[]
##tot_pred_1=0
##tot_pred_0=0
##tot_act_0=0
##tot_act_1=0
##
##for i in testY:
##    if(i[0]==1):
##        tot_act_1+=1
##    else:
##        tot_act_0+=1
##
##
for i in range(len(testX)):
    Ymy.append(clf.predict(testX[i]))
    
##for i in Ypredict:
##	if(i[0]>=0.5):
##		Ymy.append([1])
##		tot_pred_1+=1
##	else:
##		Ymy.append([0])
##		tot_pred_0+=1
##
##
pred_1_actual_1=0
pred_1_not_1=0
pred_0_actual_0=0
pred_0_not_0=0



for i in range(len(Ymy)):
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

