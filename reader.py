
from pyAudioAnalysis import audioBasicIO,audioFeatureExtraction
import operator

import numpy as np
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

baseDataset = "dataset/"
positivePath = 'PositiveRandom/'
allDataPath  = 'allPositiveData/'

indexClass=[0,1,2,3,4,5,13,14,16]

classesNames={}
classCounter=[]
c_C=0
X={}

def featureExtractor(fileName):
    [Fs, x] = audioBasicIO.readAudioFile(fileName)
    Features = audioFeatureExtraction.stFeatureExtraction(x, Fs,0.001 * Fs, 0.0009 * Fs)
    MFCCs = []
    
    for index in range(len(Features)):
        MFCCs.append(float(np.max(Features[index])))
    
    return MFCCs

def img2array(image):
    vector = []
    for line in image:
        for column in line:
            vector.append(float(column[0])/255)

    return np.array(vector)

def parseData(isImage=True):
    count = 0
    for idx, val in enumerate(os.listdir(baseDataset)):
        if not val in classesNames.keys():
            classesNames[val] = count
            count += 1

        for idx_, img_folder in enumerate(os.listdir(baseDataset + val)):
            if idx_ > 0 and os.path.isdir(baseDataset + val + '/' + img_folder):
                for marker, audio in enumerate(os.listdir(baseDataset + val + '/' + img_folder)):
                    if os.stat(baseDataset + val + '/' + img_folder +'/'+ audio).st_size==0:
                        continue

                    if marker > 0 and audio.endswith('WAV') and isImage:
                        if not classesNames[val] in X.keys():
                            X[classesNames[val]] = baseDataset + val + '/' + img_folder + '/Spec/Crop/c' + audio.replace('WAV','png')
                        else:
                            X[classesNames[val]] = X[classesNames[
                                    val]] + ',' + baseDataset + val + '/' + img_folder + '/Spec/Crop/c' + audio.replace('WAV','png')
                    elif marker > 0 and audio.endswith('WAV') and not isImage:
                        if not classesNames[val] in X.keys():
                            X[classesNames[val]] = baseDataset + val + '/' + img_folder +'/'+ audio
                        else:
                            X[classesNames[val]] = X[classesNames[
                                    val]] + ',' + baseDataset + val + '/' + img_folder +'/'+ audio

    X_ = []
    y_ = []
    sorted_classesNames= sorted(classesNames.items(), key=operator.itemgetter(0))

    realClass = 0
    for classVal in range(len(sorted_classesNames)):
        if classVal in X.keys():
            arquivos = X[classVal].split(',')

            print("accepting : ", classVal)
            for val in arquivos:
                if (isImage):
                    img_ = cv2.imread(val)
                    #img_ = img2array(img_)

                    #img_ = img_.astype('float32')

                    X_.append(img_)
                    y_.append(realClass)
                else:
                    X_.append(featureExtractor(val))
                    y_.append(realClass)
            realClass += 1

    X_ = np.array(X_)
    y_ = np.array(y_)

    return X_, y_





