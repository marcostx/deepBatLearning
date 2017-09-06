from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from pyAudioAnalysis import audioBasicIO,audioFeatureExtraction
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.misc import imread
import tensorflow as tf
import pandas as pd
from random import shuffle
import operator
import pylab
import numpy as np
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

baseDataset = "/Users/marcostexeira/Documents/codigoMorcegos/dataset/"
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

def parseData():
    count=0
    for idx, img in enumerate(os.listdir(positivePath)):
        if img.find('.png') > 0:
            #splitting name
            validImg = img[1:]
            if 'id' not in validImg:
                if validImg.split('_')[0] == 'Carollia':
                    classesName = validImg.split('_')[0] + ' '+ validImg.split('_')[1]
                    audioName   = validImg.split('_')[2].split('M')[0] + ' M'+ validImg.split('I')[0].split('M')[-1]
                    imgName     = 'c'+ validImg.split('_')[2].split('M')[0] + ' M'+ validImg.split('I')[0].split('M')[-1]
                elif not validImg.split('_')[0] == 'Stenodermatinae':
                    classesName = validImg.split('_')[0] + ' '+ validImg.split('_')[1]
                    audioName   = validImg.split('_')[2] + ' '+ validImg.split('I')[0].split('_')[-1]
                    if len(validImg.split('_'))>3:
                        imgName     = 'c'+ validImg.split('_')[2] + ' '+ validImg.split('_')[3]
                else:
                    classesName = validImg.split('_')[0]
                    audioName   = validImg.split('_')[1] + ' '+ validImg.split('I')[0].split('_')[-1]
                    imgName     = 'c'+ validImg.split('_')[1] + ' '+ validImg.split('_')[2]
                
                if not classesName in classesNames.keys() and os.path.isfile(baseDataset + classesName+'/'+audioName+'/Spec/Crop/'+imgName):
                    if count > 0:
                        classCounter.append(c_C)
                    classesNames[classesName] = count
                    #c_C=0
                    count+=1

                if os.path.isfile(baseDataset + classesName+'/'+audioName+'/Spec/Crop/'+imgName):   
                    if not classesNames[classesName] in X.keys():
                        X[classesNames[classesName]] = positivePath + img
                    else:
                        X[classesNames[classesName]] = X[classesNames[classesName]] + ',' + positivePath + img
                    

    X_=[]
    y_=[]
    
    realClass = 0
    for classVal in range(len(classesNames)):
        arquivos = X[classVal].split(',')
        
        if len(arquivos) > 12:
            #print(X[classVal].split(','))
            #print(len(arquivos))
            #print("\n\n")
            for val in arquivos:
                img_ = cv2.imread(val)
                img_ = img2array(img_)
                img_ = img_.astype('float32')

                X_.append(img_)
                y_.append(realClass)
            realClass+=1
    X_      = np.array(X_)
    y_      = np.array(y_)

    return X_,y_


def readAllAudioData(clf):
    count=0
    for idx, val in enumerate(os.listdir(allDataPath)):
        if idx > 0:
            if not val in classesNames.keys():
                classesNames[val] = count
                count+=1

            for idx_, audio_folder in enumerate(os.listdir(allDataPath + val)):
                if idx_ > 0 and os.path.isdir(allDataPath + val + '/'+ audio_folder):
                    for marker, audio in enumerate(os.listdir(allDataPath + val + '/' + audio_folder)):
                        if marker > 0 and os.path.isfile(baseDataset + val+'/'+audio_folder+'/Spec/Crop/c'+audio.replace('WAV','png')):
                            #img_ = cv2.imread(baseDataset + val+'/'+audio_folder+'/Spec/Crop/c'+audio.replace('WAV','png'))
                            #img_ = img2array(img_)

                            #pred = clf.predict(img_)
                            #if pred == [1]:
                            if not classesNames[val] in X.keys():
                                X[classesNames[val]] = baseDataset + val+'/'+audio_folder+'/'+audio.replace('png','WAV')
                            else:
                                X[classesNames[val]] = X[classesNames[val]] + ',' + baseDataset + val+'/'+audio_folder+'/'+audio.replace('png','WAV')
        
    X_=[]
    y_=[]
    sorted_classesNames = sorted(classesNames.items(), key=operator.itemgetter(0))
    
    realClass = 0
    for classVal in range(len(classesNames)):
        if classVal in X.keys():
            arquivos = X[classVal].split(',')
            
            if classVal in indexClass:
                print("accepting : ", classVal)
                for val in arquivos:
                    X_.append(featureExtractor(val))
                    y_.append(realClass)
                realClass+=1
    
    X_      = np.array(X_)
    y_      = np.array(y_)
    print(len(X_))

    return X_,y_

                        

def readAllData(clf):
    count=0
    for idx, val in enumerate(os.listdir(baseDataset)):
        if idx > 0:
            if not val in classesNames.keys():
                classesNames[val] = count
                count+=1

            for idx_, img_folder in enumerate(os.listdir(baseDataset + val)):
                if idx_ > 0 and os.path.isdir(baseDataset + val + '/'+ img_folder):
                    for marker, img in enumerate(os.listdir(baseDataset + val + '/' + img_folder)):
                        if marker > 0 and os.path.isfile(baseDataset + val+'/'+img_folder+'/Spec/Crop/c'+img.replace('WAV','png')):
                            img_real = cv2.imread(baseDataset + val+'/'+img_folder+'/Spec/Crop/c'+img.replace('WAV','png'))
                            img_ = img2array(img_real)

                            pred = clf.predict(img_)
                            if pred == 1:
                                while(1):
                                    cv2.imshow('img',img_real)
                                    k = cv2.waitKey(33)
                                    if  k == ord('q'):
                                        break
                                if not classesNames[val] in X.keys():
                                    X[classesNames[val]] = baseDataset + val+'/'+img_folder+'/Spec/Crop/c'+img.replace('WAV','png')
                                else:
                                    X[classesNames[val]] = X[classesNames[val]] + ',' + baseDataset + val+'/'+img_folder+'/Spec/Crop/c'+img.replace('WAV','png')
                            
                        
    X_=[]
    y_=[]

    realClass = 0
    for classVal in range(len(classesNames)):
        if classVal in X.keys():
            arquivos = X[classVal].split(',')
            
            if classVal in indexClass:
                print("accepting : ", classVal)
                for val in arquivos:
                    img_ = cv2.imread(val)
                    img_ = img2array(img_)

                    img_ = img_.astype('float32')
                    X_.append(img_)
                    y_.append(realClass)
                realClass+=1

    X_      = np.array(X_)
    y_      = np.array(y_)
    print(len(X_))

    return X_,y_

