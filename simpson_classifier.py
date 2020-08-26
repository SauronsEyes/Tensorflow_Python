#Simspson Classifier

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import transform
import csv
from tensorflow.keras import datasets, layers,models

results = []
results2 = []
with open("filestructure.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: 
        results.append(row)
with open("filestructure2.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: 
        results2.append(row)

print(results[0][0])
print(np.array(results).shape)
print(len(results))
a=0
labels = []
class_name = []
images = []
new=[]
uniqueCheck = ""
for i in range(len(results)):
    np_image = Image.open(results[i][1])
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (32, 32, 3))
    images.append(np_image)
    if(uniqueCheck != results[i][0]):
        a=a+1
        uniqueCheck = results[i][0]
        class_name.append(results[i][0])
    labels.append(a)
    print(str(i)+"/"+str(len(results)))
    
print(class_name)
print(len(class_name))
train_label = np.array(labels).reshape(-1,1)
train_image = np.array(images)
print(train_image.shape)

labels2=[]
images2=[]

for i in range(len(results2)):

    np_image = Image.open(results2[i][1])
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (32, 32, 3))
    images2.append(np_image)
    labels2.append(class_name.index(results2[i][0]))
    
    
test_label = np.array(labels2).reshape(-1,1)
test_image = np.array(images2)                                     
    
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(44))

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy']
              )
history  =  model.fit(train_image,train_label,epochs = 10,
                      validation_data = (test_image,test_label))
        
test_loss , test_acc = model.evaluate(test_image,test_label,verbose = 2)
print(test_acc)

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (32, 32, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('simp.jpg')



predictions = model.predict(image)
print(predictions)
plt.figure()
plt.imshow(image[0])
plt.grid(False)
plt.show()

predicted_class = class_name[np.argmax(predictions)-1]
print(predicted_class)
