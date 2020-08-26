import tensorflow as tf
from tensorflow.keras import datasets, layers,models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import transform

(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()
train_images,test_images = train_images/255.0,test_images/255.0
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10))#Number of classes airplane automobile haru

model.compile(optimizer='adam',#Optimizer function
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),#loss function
              metrics=['accuracy']
              )

history  =  model.fit(train_images,train_labels,epochs = 2,
                      validation_data = (test_images,test_labels))

test_loss , test_acc = model.evaluate(test_images,test_labels,verbose = 2)
print(test_acc)

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (32, 32, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('froggo.jpg')

print(image.shape)

predictions = model.predict(image)
print(predictions)
plt.figure()
plt.imshow(image[0])
plt.grid(False)
plt.show()

predicted_class = class_names[np.argmax(predictions)]
print(predicted_class)



