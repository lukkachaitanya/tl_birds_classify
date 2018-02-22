import keras
from keras.models import load_model
from keras.models import Sequential
import cv2
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
model = Sequential()

model =load_model('lc_auxo_birds.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('images/018.Spotted_Catbird/Spotted_Catbird_0015_3026208002.jpg')
img = cv2.resize(img,(299,299))
img = np.reshape(img,[1,299,299,3])
classes = model.predict_classes(img)
print classes

