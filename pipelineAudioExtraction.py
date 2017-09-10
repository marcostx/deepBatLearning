import backend
import sys
import os
params={
'Anoura geoffroyi': [70000, 130000, 150.0],
'Artibeus cinereus': [55000, 120000, 150.0],
'Artibeus lituratus': [50000, 130000, 150.0],
'Artibeus obscurus': [40000, 135000, 150.0],
'Artibeus planirostris': [40000, 130000, 150.0],
'Carollia perspicillata': [45000, 125000, 150.0],
'Phyllostomus hastatus': [30000, 120000, 150.0],
'Platyrrhinus helleri': [20000, 120000, 150.0],
'Pteronotus parnellii': [45000, 120000, 150.0],
}

dataset = sys.argv[1]

for directory in os.listdir(dataset):
    for filename in os.listdir(dataset + '/' + directory):
    	fmin,fmax,pmin = params[directory]

    	
        if filename == '.DS_Store':
            continue

        if (filename.find('wav') > 0 or filename.find('WAV') > 0):
            print("analisando : ", filename)
            backend.interesting_points_finder(dataset + '/' + directory + '/' + filename, fmin, fmax, pmin)
        backend.time_stamps_cropper(dataset + '/' + directory + '/' + filename)
        backend.raw_specs(dataset + '/' + directory + '/' + filename)
        backend.crop_specs(dataset + '/' + directory + '/' + filename)
