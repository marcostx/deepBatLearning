"""

	Este modulo e responsavel por realizar:
	- input: audio wav
	- quebrar em pontos de interesse
	- realizar crop nos audios
	- transformar crops em imagens
	- escolher as imagens que realmente sao sons de morcegos
	- realizar crop das imagens em grayscale
	- classificar cada uma das imagens e pegar como predicao a media
	

	* primeiro de tudo : gerar um classificador que distingua entre 
	som de morcego e nao som de morcego
"""
import sys
import csv
import os
import cv2
from scipy.misc import imread
import tensorflow as tf
import scipy
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from backend import  interesting_points_finder,time_stamps_cropper,raw_specs,crop_specs,img2array

def mainPipeline(inp):
	# do all stuff to extract the images and audios
	#interesting_points_finder(inp, 15000, 140000, 200.0)
	#time_stamps_cropper(inp)
	#raw_specs(inp)
	#crop_specs(inp)

	maindir = 'temp/' + inp.split('.')[0] + '/Spec/Crop/'
	model_path = "tempModelsCNNFolds/model6.ckpt"
	positiveImages = []
	mean_predictions= None

	# load model
	clf = joblib.load('model_classifier_positive_negative.pkl') 

	# classificar iamgens em positivas e negativas , pegar apenas as positivas para classificacao
	for fname in sorted(os.listdir(maindir)):
		image = cv2.imread(maindir + fname)
		image = img2array(image)

		pred = clf.predict(image)
		if pred == [1]:
			positiveImages.append(image.astype('float32'))

	positiveImages = np.array(positiveImages)
	# carregar modelo tensorflow
	### set all variables

	# Parameters
	learning_rate = 0.001
	training_iters = 100
	batch_size = 50
	display_step = 10

	# Network Parameters
	n_input = 56*92 # MNIST data input (img shape: 28*28)
	n_classes = 8 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units
	epochs=200

	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


	# Create some wrappers for simplicity
	def conv2d(x, W, b, strides=1):
	    # Conv2D wrapper, with bias and relu activation
	    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	    x = tf.nn.bias_add(x, b)
	    return tf.nn.relu(x)


	def maxpool2d(x, k=2):
	    # MaxPool2D wrapper
	    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
	                          padding='SAME')


	# Create model
	def conv_net(x, weights, biases, dropout):
	    # Reshape input picture
	    x = tf.reshape(x, shape=[-1, 56, 92, 1])

	    # Convolution Layer
	    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	    # Max Pooling (down-sampling)
	    conv1 = maxpool2d(conv1, k=2)

	    # Convolution Layer
	    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	    # Max Pooling (down-sampling)
	    conv2 = maxpool2d(conv2, k=2)

	    # Fully connected layer
	    # Reshape conv2 output to fit fully connected layer input
	    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	    fc1 = tf.nn.relu(fc1)
	    # Apply Dropout
	    fc1 = tf.nn.dropout(fc1, dropout)

	    # Output, class prediction
	    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	    return out

	# Store layers weight & bias
	weights = {
	    # 5x5 conv, 1 input, 32 outputs
	    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	    # 5x5 conv, 32 inputs, 64 outputs
	    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	    # fully connected, 14*23*64 inputs, 1024 outputs
	    'wd1': tf.Variable(tf.random_normal([14*23*64, 1024])),
	    # 1024 inputs, 10 outputs (class prediction)
	    'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}

	biases = {
	    'bc1': tf.Variable(tf.random_normal([32])),
	    'bc2': tf.Variable(tf.random_normal([64])),
	    'bd1': tf.Variable(tf.random_normal([1024])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Construct model
	pred = conv_net(x, weights, biases, keep_prob)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	# Initializing the variables
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	# classificar as imagens positivas

	with tf.Session() as sess:
        # create initialized variables
		sess.run(init)
        
        # Restore model weights from previously saved model
		saver.restore(sess, model_path)

		predict = tf.argmax(pred, 1)
		predictions = predict.eval({x: positiveImages.reshape(-1, n_input), keep_prob: 1.})
        
		
	# pegar a media das predicoes como predicao final
	u, indices = np.unique(predictions, return_inverse=True)
	
	print(u[np.argmax(np.bincount(indices))])

	





if __name__ == '__main__':
	mainPipeline(sys.argv[1])