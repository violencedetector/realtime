

##########################################################

#from keras.models import load_model
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_fscore_support
#import matplotlib.pyplot as plt
from sklearn import metrics

import time

#############

#DATASET=[[0,"./test2/background"],[1,"./test2/gun"]]
MODEL="3dcnn_64_test3b.model"

#dbPath="validation64_test3b.hdf5"
dbPath="Validation3DCNN.hdf5"

DEPTH=64
#############

#### codigo do gerador hdf5 verificar se funciona

#from keras.utils import np_utils
#from tensorflow.keras.utils import np_utils

import numpy as np
import h5py
from sklearn.preprocessing import LabelBinarizer##
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

		# open the HDF5 database for reading and determine the total
		# number of entries in the database
db = h5py.File(dbPath)
		##self.numImages = self.db["labels"].shape[0]
numImages = db["labels"].shape[0]
print("Num img:")
print(numImages)

# !!!!!!!!!!!!!!!!!!!criar logica de lotes pra nao estourar memoria

images = db["images"][0: 100*DEPTH]
labels = db["labels"][0: 100*DEPTH]
print(images.shape)

##############
#from hdf5 import HDF5DatasetWriter
#writer = HDF5DatasetWriter((6400,224, 224, 3), "Validation3DCNN.hdf5")
#for image_num in range(6400):
#    writer.add([images[image_num]], [int(labels[image_num])])
#writer.close()
##############

#trata 3d
# initialize the list of processed images
label_array= []	
framearray = []
long_array = []
frame_count=1
# loop over the images
for image in images:
	framearray.append(image)
	if frame_count!=DEPTH:
		frame_count+=1
	else:
		long_array.append(np.array(framearray))
		framearray = []
		frame_count=1
images = np.array(long_array)
##########
frame_count=1
for label in labels:
	if frame_count!=DEPTH:
		frame_count+=1
	else: 
		label_array.append(label)
		frame_count=1
labels = np.array(label_array)
##labels=to_categorical(label_array, 2)##	

	
model = load_model(MODEL)



print(images.shape)
print(images[0].shape)
predicted_classes_=[]

#count=0
t = time.time()

for item in range(100):
#predictions = model.predict(images, batch_size=20)

    predictions = model.predict(np.expand_dims(images[item], axis=0), batch_size=1)
    #predictions_.append(predictions)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_classes_.append(predicted_classes)
elapsed = time.time() - t
print("elapsed: "+str(elapsed/100))
#print("count: "+str(count))


print("predicted_classes")
for i in predicted_classes_:
	print(i)
print("labels")
for l in labels:
	print(l)

print("Model 3DCNN: "+MODEL)
print("confusion_matrix")
print(confusion_matrix(labels, predicted_classes_))
#plt.imshow(confusion_matrix(trainLabels, predicted_classes))
print("precision_recall_fscore_support")
#print(precision_recall_fscore_support(trainLabels, predicted_classes, average=None))
print(metrics.classification_report(labels, predicted_classes_))

#with open("trainx.npy", 'rb') as file_handle:
#    trainX = np.load(file_handle, allow_pickle=True)
#with open("trainy.npy", 'rb') as file_handle:
#    trainy = np.load(file_handle, allow_pickle=True)
with open("testx.npy", 'rb') as file_handle:
    testX = np.load(file_handle, allow_pickle=True)
with open("testy.npy", 'rb') as file_handle:
    testy = np.load(file_handle, allow_pickle=True)


print("Model LSTM: ")
model = load_model("best_acc_final.keras")
#model = load_model("LSTM_final.keras")
t = time.time()
predictions = model.predict(testX)#, batch_size=batch_size)
elapsed = time.time() - t
print("elapsed: "+str(elapsed/testX.shape[0]))
print("count: "+str(testX.shape[0]))

predicted_classes = np.argmax(predictions, axis=1)

testy = np.argmax(testy, axis=1)    

print("confusion_matrix")
print(confusion_matrix(testy, predicted_classes))
print("precision_recall_fscore_support")
print(metrics.classification_report(testy, predicted_classes))




