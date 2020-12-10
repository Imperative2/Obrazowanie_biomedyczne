import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from skimage.morphology import square
from skimage.filters import median
from skimage.feature import hog
from skimage.feature import daisy
from skimage.transform import resize
from skimage.filters import unsharp_mask
import skimage
import os




from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True)

labels = [0, 1]

#ładowanie ścieżek z folderów ze zdjęciami
train_images_normal_path = os.listdir('./archive/TrainImages/Normal')
train_images_covid_path = os.listdir('./archive/TrainImages/COVID-19')
test_images_normal_path = os.listdir("./archive/TestImages/Normal")
test_images_covid_path = os.listdir("./archive/TestImages/COVID-19")

print(len(train_images_normal_path))
print(len(train_images_covid_path))
print(len(test_images_normal_path))
print(len(test_images_covid_path))

#tworzenie list ze zdjęciami i eytkietami
X_train = []
X_test = []

y_train = []
y_test = []
i =0

#ładowanie zdjęć do list
for path in train_images_normal_path:
    im = np.array(image.imread('./archive/TrainImages/Normal/'+path))
    if im.ndim == 3:
        dim = im.shape[2]
        im = np.sum(im, axis=2)
        im = im/((2*8)**dim)
    im = (im * 255).astype(np.uint8)
    X_train.append(im)
    y_train.append(labels[0])
for path in train_images_covid_path:
    im = np.array(image.imread('./archive/TrainImages/COVID-19/'+path))
    if im.ndim == 3:
        dim = im.shape[2]
        im = np.sum(im, axis=2)
        im = im/((2*8)**dim)
    im = (im * 255).astype(np.uint8)
    X_train.append(im)
    y_train.append(labels[1])
for path in test_images_normal_path:
    im = np.array(image.imread('./archive/TestImages/Normal/'+path))
    if im.ndim == 3:
        dim = im.shape[2]
        im = np.sum(im, axis=2)
        im = im/((2*8)**dim)
    im = (im * 255).astype(np.uint8)
    X_test.append(im)
    y_test.append(labels[0])
for path in test_images_covid_path:
    im = np.array(image.imread("./archive/TestImages/COVID-19/"+path))
    if im.ndim == 3:
        dim = im.shape[2]
        im = np.sum(im, axis=2)
        im = im/((2*8)**dim)
    im = (im * 255).astype(np.uint8)
    X_test.append(im)
    y_test.append(labels[1])


print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))





#szukanie najmniejszej wysokości i szerokości
min_height = X_train[0].shape[0]
for image in X_train:
    if image.shape[0] < min_height:
        min_height = image.shape[0]
for image in X_test:
    if image.shape[0] < min_height:
        min_height = image.shape[0]
print("min height: ",min_height)

min_width = X_train[0].shape[1]
for image in X_train:
    if image.shape[1] < min_width:
        min_width = image.shape[1]
for image in X_test:
    if image.shape[1] < min_width:
        min_width = image.shape[1]
print("min width: ",min_width)

print("starting processing")
X_train_processed = np.zeros(shape=(len(X_train),min_height,min_width), dtype="uint8")
X_test_processed = np.zeros(shape=(len(X_test),min_height,min_width), dtype="uint8")


for i in range(len(X_train)):
    print(i)
    scaled = resize(X_train[i], output_shape=(min_height, min_width))
    scaled = unsharp_mask(scaled, radius=1, amount=1)
    scaled = (scaled*255).astype("uint8")
    print(scaled)
    X_train_processed[i] = scaled
for i in range(len(X_test)):
    print(i)
    scaled = resize(X_test[i], output_shape=(min_height, min_width))
    scaled = unsharp_mask(scaled, radius=1, amount=1)
    scaled = (scaled*255).astype("uint8")
    print(scaled)
    X_test_processed[i] = scaled

print("finished processing")

fig, ax = plt.subplots(2, 2, figsize=(10,20))
ax = ax.ravel()

print(X_train_processed[289])

ax[0].imshow(X_train[289])
ax[1].imshow(X_train_processed[289])
ax[2].imshow(X_test[0])
ax[3].imshow(X_test_processed[0])
plt.show()




