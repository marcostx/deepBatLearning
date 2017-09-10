

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from random import shuffle
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from reader import parseData
import time

#clf = joblib.load('model_classifier_positive_negative.pkl') 
#X_, y_multiclass = readAllAudioData()
X_, y_multiclass = parseData(isImage=False)

acc_vals	= []

#X_ = np.array(X_)
#y_multiclass = np.array(y_multiclass)
precisions,accuracies,recalls,f1s = [],[],[],[]
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

	accuracies.append(accuracy_score(y_test, pred))
	precisions.append(precision_score(y_test, pred, average='weighted'))
	recalls.append(recall_score(y_test, pred, average='weighted'))
	f1s.append(f1_score(y_test, pred, average='weighted'))

	print("accuracy : ", accuracy_score(y_test, pred ) )
	print("precision : ", precision_score(y_test, pred, average='weighted' ) )
	print("recall : ", recall_score(y_test, pred,average='weighted' ) )
	print("f1 : ", f1_score(y_test, pred,average='weighted' ) )
	print("\n")

print("Done.")

print("accuracy avg : ", np.mean(accuracies) )
print("precision avg : ", np.mean(precisions) )
print("recall avg : ", np.mean(recalls) )
print("f1 avg : ", np.mean(f1s) )

print("it took", time.time() - start, "seconds.")