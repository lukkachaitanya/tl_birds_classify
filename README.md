
>### *“To iterate is human, to recurse divine.”* 



# Birds Classify :   *Using Transfer Learning on InceptionV3*
Download dataset from [here](http://www.vision.caltech.edu/visipedia/CUB-200.html)
#### Dataset description
Caltech-UCSD Birds 200 (CUB-200) is an image dataset with photos of 200 bird species (mostly North American). 

    Number of categories: 200

    Number of images: 6,033



## Getting started
The goal of this project is to use transfer learning and fine-tuning to identify any bird classes from the given dataset.
 
 We follow the below two steps in order:

 - **Transfer learning**: Let us take a ConvNet that has been pre-trained on ImageNet, remove the last fully-connected layer, then treat the rest of the ConvNet as a feature extractor for the new dataset. Once we extract the features for all images, then train a classifier for the new dataset.
 - **Fine-tuning**: Now replace and retrain the classifier on top of the ConvNet, and also fine-tune the weights of the pre-trained network via backpropagation.

    



## Software to pre-install
 This is supported for Python >= 2.7 and Python >= 3.3.

**Dependencies :**
```
Pillow==5.0.0
numpy==1.14.0
Tensorflow==1.5.0
Keras==2.1.3

```
# Approach
Please follow the below steps.
## Steps:
### Get data

```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
tar -xvzf images.tar.gz
```
Images folder contains 200 categories of Birds dataset. With a total of `6033`  images.

![enter image description here](https://imgur.com/gkvS1Bj.png)
### Split data:

 - We then split the data into **train/test**  folders in the ratio of `70:30`. i.e `70%` of images in training set and `30%` of images in validation set.
 - We can also divide the data into **train/valid/test** in the ratio `60:20:20` but here, in this implementation we follow the former approach i.e `70:30`
 
```
	import os, sys
	import shutil
	import random
	from shutil import copyfile


	source_dir = 'images/'
	for root, dirs, files in os.walk(source_dir):
	    for i in dirs: 
	        
	        path = 'test/' + "%s/" % i
	        os.makedirs(path)

	        filenames = random.sample(os.listdir('images/' + "%s/" % i ), int(len(os.listdir('images/' + "%s/" % i ))*0.3))
	        for j in filenames:
	            shutil.move('images/' + "%s/" % i  + j, path)

```

 - We use the above simple script to recursively copy and randomly move the `30%` of the images of each class into our **test** directory.
	 ```
	 filenames = random.sample(os.listdir('images/' + "%s/" % i ), int(len(os.listdir('images/' + "%s/" % i ))*0.3))
	 ```

 - This above line helps use in randomly selecting `30%` of the images in each folder i.e `from 200 classes`. You can find above script in `split.py`

- Use 
```find train -type f | wc -l```  and  ```find test -type f | wc -l``` 
to count the no. of files in train and test folders and verify the splitting process.  We get ``` 4318```   and ```1715``` respectively.


----------
### Imports:
Call all necessary libraries and system funcs:
		

    import os
	import sys
	import glob
	import argparse
	import matplotlib.pyplot as plt

	from keras import __version__
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	from keras.models import Model
	from keras.layers import Dense, GlobalAveragePooling2D
	from keras.preprocessing.image import ImageDataGenerator
	from keras.optimizers import SGD
	from keras.utils import plot_model


----------
### Declare all global variables:
```
IM_WIDTH, IM_HEIGHT = 299, 299 
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

train_dir= 'train' # locate training folder
val_dir='test' 	   # locate test folder
nb_epoch=NB_EPOCHS
batch_size=BAT_SIZE
```


----------


### Data Augmentation:

> Data augmentation is the process of artificially increasing the size
> of your dataset via transformations.

	

    # data augementation
	  train_datagen =  ImageDataGenerator(
	      preprocessing_function=preprocess_input,
	      rotation_range=30,
	      width_shift_range=0.2,
	      height_shift_range=0.2,
	      shear_range=0.2,
	      zoom_range=0.2,
	      horizontal_flip=True
	  )
	  test_datagen = ImageDataGenerator(
	      preprocessing_function=preprocess_input,
	      rotation_range=30,
	      width_shift_range=0.2,
	      height_shift_range=0.2,
	      shear_range=0.2,
	      zoom_range=0.2,
	      horizontal_flip=True
	  )

	  train_generator = train_datagen.flow_from_directory(
	    train_dir,
	    target_size=(IM_WIDTH, IM_HEIGHT),
	    batch_size=batch_size,
	  )

	  validation_generator = test_datagen.flow_from_directory(
	    val_dir,
	    target_size=(IM_WIDTH, IM_HEIGHT),
	    batch_size=batch_size,
	  )


----------

### Define base model:
```
base_model = InceptionV3(weights='imagenet', include_top=False) 
#include_top=False excludes final FC layer

model = add_new_last_layer(base_model, nb_classes)

```

### Call to Transfer learning func:

 - Here we are using ```rmsprop``` optimizer we can also try ```adam``` and check the results accuracy.

```
def setup_to_transfer_learn(model, base_model):

  for layer in base_model.layers:
    layer.trainable = False
  model.summary()
  plot_model(model, to_file='model.png')
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

```

- ```model.summary()```  helps us in analyzing the model and check the parameters in each layer.

> **Note**: All layers are not displayed below.  Just for intuition purpose only.

![enter image description here](https://imgur.com/ynhMqL1.png)
 
----------
### Call to fine-tuning func:

> Here we are using standard learning rate and momentum parameters. i.e
> `lr=0.0001` and `momentum=0.9`

```
def setup_to_finetune(model):
  """Freeze the bottom 172 and retrain the remaining top layers.

  for layer in model.layers[:172]:
     layer.trainable = False
  for layer in model.layers[172:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
```

### Fit Model: using model.fit_generator()

> Fits the model on data generated batch-by-batch by a Python generator.
> The generator is run in parallel to the model, for efficiency.

For Transfer Learning:
```
  history_tl = model.fit_generator(
    train_generator,
    epochs=nb_epoch,
    steps_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    validation_steps=nb_val_samples,
    class_weight='auto')

```

Then for Fine-tuning:
```
  history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=nb_val_samples,
    class_weight='auto')

```

### Training:
Call `train()` to start the training process.
Once the process is started we can see an output as below:
![enter image description here](https://i.imgur.com/R4rG6dS.jpg)

----------
### Finally *Save* the model:
Do not forget to  save the model, you can use:
```model.save('lc_auxo_birds.h5')```


### Plotting the Cost vs Epochs:
Use the `plot_training()` func to plot the 

```

def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()

```



### Using the saved model:
```
import keras
from keras.models import load_model
from keras.models import Sequential
import cv2
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
model = Sequential()

model =load_model('lc_auxo_birds.h5') ##load the saved model in the previous step
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('images/018.Spotted_Catbird/Spotted_Catbird_0015_3026208002.jpg')
img = cv2.resize(img,(299,299))
img = np.reshape(img,[1,299,299,3])
classes = model.predict_classes(img)
print classes
```
```model.predict_classes()``` outputs the predicted image category.


----------
### Command line usage:
```
sudo python3 auxo_tl.py --train_dir train --val_dir test
```



### Future scope to improve performance:

 - We may use L2 regularizer to improve the performance.
 - Can also add Batch normalization layer to avoid the overfitting.
 - We use gradient checking to evaluate the back propagation.


----------


### Screencast:

[![Screencast](https://imgur.com/5t0okF7.png)](https://www.youtube.com/watch?v=P6te0ZEPlBw "Transfer Learning on Inceptionv3")

## License

[MIT License]


