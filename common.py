import sys
import os
import cv2


def img2array(image):
    vector = []
    for line in image:
        for column in line:
            vector.append("%4.3f"%(float(column[0])/255))
    return vector

def save_file(csvName, imageArray, labelsArray):
    csvFile = open(csvName,'w')
    iterator = 0
    for line in imageArray:
        csvFile.write(','.join(line) + ',%s\n'%(labelsArray[iterator]))
        iterator+=1
    csvFile.close()

def png_to_csv(imagesdir):
	imagesdir = imagesdir

	imageArray = []
	labelsArray = []
	for fileName in sorted(os.listdir(imagesdir+'NegativeRandom/')):
		if fileName[0] == 'c':
			label = -1
			labelsArray.append(label)
			figname = imagesdir+'NegativeRandom/' + os.sep + fileName
			image = cv2.imread(figname)
			imageArray.append(img2array(image))
	for fileName in sorted(os.listdir(imagesdir+'/PositiveRandom')):
		if fileName[0] == 'c':
			label = 1
			labelsArray.append(label)
			figname = imagesdir+'PositiveRandom/' + os.sep + fileName
			image = cv2.imread(figname)
			imageArray.append(img2array(image))

	csvName = imagesdir + 'result.csv'
	print 'Local do arquivo destino: ' + csvName
	save_file(csvName, imageArray, labelsArray)


png_to_csv(sys.argv[1])
	