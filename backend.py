

import warnings #TODO corrigir future warning
import wave
import sys
import os
import numpy as np
import pylab
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
warnings.filterwarnings("ignore") #TODO corrigir future warning

def interesting_points_finder(filestring, freq_min, freq_max, pot_min):

   nomewav = os.path.basename(filestring)
   filename = os.path.splitext(nomewav)[0]
   
   if not os.path.exists("temp/"+filename+"/"):
	   os.makedirs("temp/"+filename+"/")
   
   win = wave.open(filestring, 'r')

   sampFreq = win.getframerate()
   windowsize = 0.001 #tamanho da janela
   audiolen = int(windowsize * sampFreq)
   timelapse = 0.000
   time_ms = win.getnframes()*1000/win.getframerate()
   timeStamps = []

   while int((timelapse+0.001) * sampFreq) < win.getnframes():
      win.readframes(int(timelapse * sampFreq))
      snd = np.fromstring(win.readframes(audiolen), dtype=np.int16)

      n = len(snd)
      p = np.fft.fft(snd)

      nUniquePts = int(math.ceil((n+1)/2.0))
      p = p[0:nUniquePts]
      #p = abs(p)
      p = p / float(n)
      p = p**2

      if n % 2 > 0:
         p[1:len(p)] = p[1:len(p)] * 2
      else:
         p[1:len(p) -1] = p[1:len(p) - 1] * 2

      freqArray = np.arange(0, nUniquePts, 1.0) * (sampFreq / n);

      count = 0
      out_range_count = 0

      for i in range(0, nUniquePts):
         if freqArray[i] > int(freq_min) and freqArray[i] < int(freq_max): #range de Frenquencias
            if abs(p[i]) > float(pot_min): #range de decibeis
               #print "(time_b: %f), (duration: %f), (p: %f), (freq: %d)" %(timelapse, windowsize, abs(p[i]), freqArray[i])
               count = count + 1
         else:
            if abs(p[i]) > float(pot_min):
               out_range_count = out_range_count + 1

      if count > 10 and ((count + out_range_count) < (nUniquePts - 2)):
         # storing the initial value of frequency
         timeStamps.append(float("{0:.4f}".format(timelapse)))
        

      timelapse+= 0.001
      frequenciasTemporarias=[]
      potenciasTemporarias=[]
      win.close()
      win = wave.open(filestring, 'rb')

   win.close() 

   timestampsfile = open("temp/"+filename+"/"+filename+'Timestamps.txt','w')
   timestampsfile.write(str(timeStamps)) 
   timestampsfile.close()
   

def time_stamps_cropper(filestring):
   import sys
   import os
   import wave

   if (filestring.find('wav') > 0 or filestring.find('WAV') > 0):
      nomewav = os.path.basename(filestring)
      filename = os.path.splitext(nomewav)[0]

      fname = open('temp/'+filename+'/'+filename+'Timestamps.txt').read().split(', ')
      fname[0] = fname[0].translate(None, '[')
      fname[len(fname)-1] = fname[len(fname)-1].translate(None, ']\n')
      timestamps = []
         
      stop = 0
      while stop < len(fname):
         count = 0
         if stop < len(fname)-2:
            while int(float(fname[stop])*1000) + 1 == int(float(fname[stop+1])*1000) and stop+1 < len(fname)-1:
               count = count + 1
               stop = stop + 1
            if int(float(fname[stop])*1000) + 1 == int(float(fname[stop+1])*1000) and stop+1 == len(fname)-1:
               count = count + 1
               stop = stop + 1
         if count > 0:
            index = int((count)/2)
            timestamps.append(float(fname[stop-index]))
         else:
            timestamps.append(float(fname[stop]))
         stop = stop + 1
      #print timestamps


      for i in range(0, len(timestamps)):
         #print timestamps[i]
         win = wave.open(filestring, 'rb')
         wout = wave.open("temp/"+filename+"/"+filename+"I"+str(i)+".WAV", 'wb')
         timelapse = timestamps[i] - 0.003
         if timelapse > 0.0:
            win.readframes(int(timelapse * win.getframerate()))
            cropedwavframes = win.readframes(int(0.006 * win.getframerate()))
            wout.setparams(win.getparams())
            wout.writeframes(cropedwavframes)
            wout.close()
            win.close()
      print "Audios das timestamps extraidos"
   
def raw_specs(filestring):
   from scikits.audiolab import wavread
   import pylab
   import matplotlib.pyplot as plt
   import sys
   import os

   if (filestring.find('wav') > 0 or filestring.find('WAV') > 0):
      nomewav = os.path.basename(filestring)
      filename = os.path.splitext(nomewav)[0]

      maindir = "temp/"+filename+"/"
      for fnamefiles in os.listdir(maindir):
         if os.path.isdir(maindir + fnamefiles) or os.stat(maindir+fnamefiles).st_size == 0:
               print "not a file."
         else:
            if fnamefiles.find('wav') > 0 or fnamefiles.find('WAV') > 0:
               if not os.path.exists(maindir+"/Spec/"):
                  os.makedirs(maindir+"/Spec/")

               signal, fs, enc = wavread(maindir+fnamefiles);

               NFFT = 256     # the length of the windowing segments
               Fs = int(300)  # the sampling frequency

               pylab.figure(num=None, figsize=(4, 8),frameon=False)
               Pxx, freqs, bins, im = pylab.specgram(signal, NFFT=NFFT, Fs=Fs, noverlap=int(NFFT-1),cmap=pylab.cm.gist_heat)  
               if fnamefiles.find('wav') > 0:
                  figname = maindir+"/Spec/"+os.sep+fnamefiles.replace('wav','png') 
               else:
                  figname = maindir+"/Spec/"+os.sep+fnamefiles.replace('WAV','png')
               pylab.savefig(figname)
               plt.close('all')        

      print "Spectrogramas gerados."   
   
def crop_specs(filestring):
   import sys
   import os
   import cv2

   
   nomewav = os.path.basename(filestring)
   filename = os.path.splitext(nomewav)[0]

   maindir = "temp/"+filename+"/Spec/"

   for fname in sorted(os.listdir(maindir)):
       if fname.find('png') > 0 and fname[0] != 'c':
           figname = maindir+os.sep+fname
           img = cv2.imread(figname, 0)
           crop_img = img[140:600,50:330]
           tdim = crop_img.shape
           c = 5.0
           tx = int(tdim[0]/c)
           ty = int(tdim[1]/c)
           imgr = cv2.resize(crop_img,(ty,tx),interpolation=cv2.INTER_AREA)

           if not os.path.exists(maindir+"/Crop/"):
               os.makedirs(maindir+"/Crop/")

           figname = maindir+"/Crop/"+os.sep+"c"+fname
           cv2.imwrite(figname,imgr)
   print "Crop realizado."

def img2array(image):
    vector = []
    for line in image:
        for column in line:
            vector.append("%4.3f"%(float(column[0])/255))
    return np.array(vector)

#interesting_points_enconter(sys.argv[1], 15000, 140000, 200.0)
#time_stamps_cropper(sys.argv[1])
#raw_specs(sys.argv[1])
#crop_specs(sys.argv[1])


