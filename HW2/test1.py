# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:20:47 2018

@author: Administrator
"""

import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from PIL import Image

from foolbox.criteria import Misclassification
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)
# get source image and label
image = Image.open('C:/Users/Administrator/Desktop/n01844917/n01844917_497.jpeg')
shape=(224,224)
image=image.resize(shape)
image = np.asarray(image, dtype=np.float32)
image = image[:, :, :3]
assert image.shape == shape + (3,)
label = np.argmax(fmodel.predictions(image))
criterion = Misclassification()
# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.ContrastReductionAttack(fmodel)
adversarial = attack(image[:, :, ::-1], label)
import matplotlib.pyplot as plt

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()

ad_label=np.argmax(fmodel.predictions(adversarial))
print("original_label is:",label)
print("adversarial_label is:",ad_label)
print(foolbox.utils.softmax(fmodel.predictions(adversarial))[ad_label])
adversarial_rgb = adversarial[np.newaxis, :, :, ::-1]
preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
print("Top 3 predictions (adversarial: ", decode_predictions(preds, top=3))