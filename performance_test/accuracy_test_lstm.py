

##########################################################

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import time

#############
MODEL="3dcnn_64_test3b.model"

dbPath="Validation3DCNN.hdf5"

DEPTH=64
#############

import numpy as np
import h5py
from sklearn.preprocessing import LabelBinarizer##
from tensorflow.keras.utils import to_categorical

# open the HDF5 database for reading and determine the total
# number of entries in the database
db = h5py.File(dbPath)
numImages = db["labels"].shape[0]
print("Num img:")
print(numImages)

images = db["images"][0: 100*DEPTH]
labels = db["labels"][0: 100*DEPTH]
print(images.shape)


#3D preparation
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
	
model = load_model(MODEL)

predicted_classes_=[]

t = time.time()

for item in range(100):
    predictions = model.predict(np.expand_dims(images[item], axis=0), batch_size=1)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_classes_.append(predicted_classes)
elapsed = time.time() - t
print("elapsed: "+str(elapsed/100))


print("Model 3DCNN: "+MODEL)
print("confusion_matrix")
print(confusion_matrix(labels, predicted_classes_))
print("precision_recall_fscore_support")
print(metrics.classification_report(labels, predicted_classes_))

#LSTM
with open("testx.npy", 'rb') as file_handle:
    testX = np.load(file_handle, allow_pickle=True)
with open("testy.npy", 'rb') as file_handle:
    testy = np.load(file_handle, allow_pickle=True)

print("Model LSTM: ")
model = load_model("best_acc_final.keras")

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




