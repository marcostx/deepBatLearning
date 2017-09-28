# Simple CNN model for bat species classification
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import time
import keras
from random import shuffle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
K.set_image_dim_ordering('th')

from reader import parseData

def createCNNModel(num_classes):
    """ Adapted from: # http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# """
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(92, 56, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 120
    lrate = 0.004
    decay = lrate/epochs
    sgd = Adam(lr=lrate, epsilon=1e-08, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())
    return model, epochs


X,y = parseData(isImage=True)

precisions,accuracies,recalls,f1s = [],[],[],[]

skf = StratifiedKFold(n_splits=10, shuffle=True)
print("Training")
counter=0
start = time.time()
for train_index, test_index in skf.split(X, y):
	print("Fold : ",counter)

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]


	# normalize inputs from 0-255 and 0.0-1.0
	X_train = np.array(X_train).astype('float32')
	X_test = np.array(X_test).astype('float32')
	X_train = X_train / 255.0
	X_test = X_test / 255.0

	# one hot encode outputs
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]

	# create our CNN model
	model, epochs = createCNNModel(num_classes)
	print("CNN Model created.")
	# fit and run our model
	seed = 7
	np.random.seed(seed)
	model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=50)
	# Final evaluation of the model
	pred = model.predict(X_test, verbose=0)
	pred = [np.argmax(item) for item in pred]
	y_test = y[test_index]

	accuracies.append(accuracy_score(y_test, pred))
	precisions.append(precision_score(y_test, pred, average='weighted'))
	recalls.append(recall_score(y_test, pred, average='weighted'))
	f1s.append(f1_score(y_test, pred, average='weighted'))
	print("accuracy : ", accuracy_score(y_test, pred ) )
	print("precision : ", precision_score(y_test, pred, average='weighted' ) )
	print("recall : ", recall_score(y_test, pred,average='weighted' ) )
	print("f1 : ", f1_score(y_test, pred,average='weighted' ) )
	print("\n")

	text_file = open('output.txt','a')
	txt ="accuracy : " + str(accuracy_score(y_test, pred ))  + "\n" + "precision : " + str(precision_score(y_test, pred, average='weighted' ))  + "\n " + "recall : " + str(recall_score(y_test, pred,average='weighted' ))  + ' \n ' +  "f1 : " + str(f1_score(y_test, pred,average='weighted' ))  + ' \n\n '
	
	text_file.write(txt)
	text_file.close()
	counter+=1

print("Done.")

print("accuracy avg : ", np.mean(accuracies) )
print("precision avg : ", np.mean(precisions) )
print("recall avg : ", np.mean(recalls) )
print("f1 avg : ", np.mean(f1s) )


