import backend
import sys
import os

dataset = sys.argv[1]

for directory in os.listdir(dataset):
    for filename in os.listdir(dataset + '/' + directory):
        if filename == '.DS_Store':
            continue

        if (filename.find('wav') > 0 or filename.find('WAV') > 0):
            print("analisando : ", filename)
            backend.interesting_points_finder(dataset + '/' + directory + '/' + filename, 55000, 120000, 200.0)
        backend.time_stamps_cropper(dataset + '/' + directory + '/' + filename)
        backend.raw_specs(dataset + '/' + directory + '/' + filename)
        backend.crop_specs(dataset + '/' + directory + '/' + filename)
