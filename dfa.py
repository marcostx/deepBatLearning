#passos para realizar os experimentos

# ler as pastas PositiveRandom e NegativeRandom
# para cada imagem lida, se a imagem existir dentro de dataset, ler o arquivo correspondente
# obter as features do arquivo correspondente usando fft (freq_max, freq_min, duracao, pot_max)
# guardar essas features em um dicionario referenciado pelo nome da pasta
# usar as features para treinar um dfa
# usar as imagens lidas para treinar uma rede convolucional
# execucao : python main.py PositiveRandom

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from pyAudioAnalysis import audioBasicIO,audioFeatureExtraction
from random import shuffle
import numpy as np
import re
import csv
import os	
import sys
import math
import warnings #TODO corrigir future warning
import wave
import scipy.io.wavfile as wavfile
import warnings
from sklearn.externals import joblib
warnings.filterwarnings('ignore')
from reader import readAllAudioData, featureExtractor
import time
from glob import glob
from os.path import basename, join, exists

#clf = joblib.load('model_classifier_positive_negative.pkl') 
X_, y_multiclass = readAllAudioData()

acc_vals	= []

#X_ = np.array(X_)
#y_multiclass = np.array(y_multiclass)

skf = StratifiedKFold(n_splits=10, shuffle=True)
print("Training")
start = time.time()
for train_index, test_index in skf.split(X_, y_multiclass):
	shuffle(test_index)
	shuffle(train_index)

	X_train, X_test = X_[train_index], X_[test_index]
	y_train, y_test = y_multiclass[train_index], y_multiclass[test_index]

	clf = OneVsRestClassifier(LinearDiscriminantAnalysis()).fit(X_train, y_train)
	
	pred = clf.predict(X_test)
	acc_vals.append(accuracy_score(y_test, pred ))
	
	print("accuracy : ", accuracy_score(y_test, pred ) )
	print("precision : ", precision_score(y_test, pred, average='weighted' ) )
	print("recall : ", recall_score(y_test, pred,average='weighted' ) )
	print("f1 : ", f1_score(y_test, pred,average='weighted' ) )
	print("\n")

print("Done.")
print("it took", time.time() - start, "seconds.")