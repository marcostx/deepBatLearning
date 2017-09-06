from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.misc import imread
import tensorflow as tf
import pandas as pd
from random import shuffle
import pylab
import numpy as np
import os
import cv2
from reader import parseData, readAllData, readAllAudioData
from sklearn.externals import joblib
import time

seed = 128
rng = np.random.RandomState(seed)

clf = joblib.load('model_classifier_positive_negative.pkl') 
X_, y_ = readAllData(clf)

def dense_to_one_hot(labels_dense, num_classes=9):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    #index_offset = np.arange(num_labels) * num_classes
    #labels_one_hot = np.zeros((num_labels, num_classes))
    #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, y):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    batch_x = eval('train_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    batch_y = y[[batch_mask]]
    batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y

### set all variables

# number of neurons in each layer
input_num_units = 56*92
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
output_num_units = 9

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 100
batch_size = 50
learning_rate = 0.001

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([input_num_units, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, output_num_units]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([output_num_units]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

skf = StratifiedKFold(n_splits=10, shuffle=True)
number = 0
start = time.time()
for train_index, test_index in skf.split(X_, y_):
    shuffle(test_index)
    shuffle(train_index)

    X_train, X_test = X_[train_index], X_[test_index]
    y_train, y_test = y_[train_index], y_[test_index]
    
    train_x = np.stack(X_train)
    print("Fold : ", number)

    with tf.Session() as sess:
        # create initialized variables
        sess.run(init)
        
        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(len(train_x)/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], y_train)
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                
                avg_cost += c / total_batch
                
            #print "Epoch:", (epoch+1), "cost =", "{:.10f}".format(avg_cost)
        
        print "\nTraining complete!"
    

        # find predictions on val set
        predict = tf.argmax(pred, 1)
        predictions = predict.eval({x: X_test.reshape(-1, input_num_units)})
        
        print("accuracy : ", accuracy_score(y_test, predictions ) )
        print("precision : ", precision_score(y_test, predictions, average='weighted' ) )
        print("recall : ", recall_score(y_test, predictions,average='weighted' ) )
        print("f1 : ", f1_score(y_test, predictions,average='weighted' ) )
        print("\n")
    
    number+=1

print("it took", time.time() - start, "seconds.")