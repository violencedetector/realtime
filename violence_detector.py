#################################################################
#VIOLENCE DETECTOR PROJECT MORE INFO AT violencedetector.org
#Usage python3 violence_detector.py
#################################################################

from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np

import argparse

##################################################################
#			CLASS DEFINITION			
##################################################################		
class violence_detector:
	
	def __init__(self):
		self.model = MobileNet(weights='imagenet')
		
	def preprocess(self,img):
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)		
		return preprocess_input(img)

	def predict(self,img):
		preds = self.model.predict(img)
		return decode_predictions(preds, top=3)[0]
		
	def decision(self, probabilities):	
		for classid,desc,probs in probabilities:
			if classid in ('n04086273','n02749479','n04090263'):
				return True
			return False

##################################################################
#			MAIN CODE BLOCK
##################################################################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to image")
args = ap.parse_args()

img_path = args.image
img = image.load_img(img_path, target_size=(224, 224))

ViolenceDetector = violence_detector()
preprocessed = ViolenceDetector.preprocess(img)
probabilities = ViolenceDetector.predict(preprocessed)
print(ViolenceDetector.decision(probabilities))



