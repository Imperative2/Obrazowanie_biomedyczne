import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.svm import LinearSVC
from skimage.feature import ORB, match_descriptors
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.morphology import square
from skimage.filters import median
from skimage.feature import hog
from skimage.feature import daisy
from skimage.transform import resize
from skimage.filters import unsharp_mask
import skimage
import os
from skimage.feature import multiblock_lbp

from PIL import Image



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
print("loading pictures into memory")
for path in train_images_normal_path:
    im = np.asarray(Image.open('./archive/TrainImages/Normal/'+path))
    if im.ndim == 3:
        dim = 3
        im = np.sum(im, axis=2).astype(np.float)
        im = im/((2*8)**dim)
        im = (im * 255).astype(np.uint8)
    X_train.append(im)
    y_train.append(labels[0])
for path in train_images_covid_path:
    im = np.asarray(Image.open('./archive/TrainImages/COVID-19/'+path))
    if im.ndim == 3:
        dim = 3
        im = np.sum(im, axis=2).astype(np.float)
        im = im/((2*8)**dim)
        im = (im * 255).astype(np.uint8)
    X_train.append(im)
    y_train.append(labels[1])
for path in test_images_normal_path:
    im = np.asarray(Image.open('./archive/TestImages/Normal/'+path))
    if im.ndim == 3:
        dim = 3
        im = np.sum(im, axis=2).astype(np.float)
        im = im/((2*8)**dim)
        im = (im * 255).astype(np.uint8)
    X_test.append(im)
    y_test.append(labels[0])
for path in test_images_covid_path:
    im = np.asarray(Image.open("./archive/TestImages/COVID-19/"+path))
    if im.ndim == 3:
        dim =3
        im = np.sum(im, axis=2).astype(np.float)
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

X_train_all = np.zeros(shape=(3, len(X_train), min_height, min_width), dtype="uint8")
X_test_all = np.zeros(shape=(3, len(X_test), min_height, min_width), dtype="uint8")

for i in range(X_train_all.shape[1]):
    print(i)
    scaled = resize(X_train[i], output_shape=(min_height, min_width))

    X_train_all[0][i] = (scaled * 255).astype("uint8")

    sharpness = unsharp_mask(scaled, radius=1, amount=1)
    X_train_all[1][i] = (sharpness * 255).astype("uint8")

    med = median(scaled, square(3))
    X_train_all[2][i] = (med * 255).astype("uint8")

for i in range(X_test_all.shape[1]):
    print(i)
    scaled = resize(X_test[i], output_shape=(min_height, min_width))

    X_test_all[0][i] = (scaled * 255).astype("uint8")

    sharpness = unsharp_mask(scaled, radius=1, amount=1)
    X_test_all[1][i] = (sharpness * 255).astype("uint8")

    med = median(scaled, square(3))
    X_test_all[2][i] = (med * 255).astype("uint8")




for i in range(X_train_all.shape[0]):
    extractors_count = 5
    X_train_processed = X_train_all[i]
    X_test_processed = X_test_all[i]
    train_features_extracted = [[] * extractors_count]
    test_features_extracted = [[] * extractors_count]
    print(test_features_extracted)
    for img in X_train_processed:

        # DAISY
        descs = daisy(img, step=int(img.shape[0] / 2), radius=int(img.shape[1] / 5), rings=2, histograms=8,
                             orientations=8, visualize=False)
        train_features_extracted[0].append(np.asarray(descs).reshape(-1))

        #HOG
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(int(img.shape[0]/10), int(img.shape[1]/10)),
                            cells_per_block=(1,1), visualize=True)

        #ORB
        orbDetector = ORB(n_keypoints=500, fast_threshold=0.08)
        res = orbDetector.detect_and_extract(img).descriptors
        print(res)


        # print(np.asarray(fd).shape)
        # print(fd)
        fig, ax = plt.subplots()
        ax.imshow(res)
        plt.show()


print("finished processing")

#experyment nr 1

print("finding features")
orb_train_extracted = []
orb_test_extracted = []

detector_extractor1 = ORB(n_keypoints=50, fast_threshold=0.01)


daisy_train_extr = []
daisy_test_extr = []
daisy_train_extr_proccessed = []
daisy_test_extr_proccessed = []



for img in X_train:
    descs, descs_img = daisy(img, step=int(img.shape[0]/2), radius=int(img.shape[1]/5), rings=2, histograms=8,
                             orientations=8, visualize=True)
    daisy_train_extr.append(np.asarray(descs).reshape(-1))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(descs_img)
    descs_num = descs.shape[0] * descs.shape[1]
    ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    plt.show()



for img in X_test:
    descs, descs_img = daisy(img, step=int(img.shape[0] / 2), radius=int(img.shape[1] / 5), rings=2, histograms=8,
                             orientations=8, visualize=True)

    daisy_test_extr.append(np.asarray(descs).reshape(-1))


for img in X_train_processed:
    descs, descs_img = daisy(img, step=int(img.shape[0]/2), radius=int(img.shape[1]/5), rings=2, histograms=8,
                             orientations=8, visualize=True)
    daisy_train_extr_proccessed.append(np.asarray(descs).reshape(-1))



for img in X_test_processed:
    descs, descs_img = daisy(img, step=int(img.shape[0] / 2), radius=int(img.shape[1] / 5), rings=2, histograms=8,
                             orientations=8, visualize=True)

    daisy_test_extr_proccessed.append(np.asarray(descs).reshape(-1))


print("training")
svm_raw = LinearSVC()
svm_proccessed = LinearSVC()

svm_raw.fit(daisy_train_extr, y_train)
svm_proccessed.fit(daisy_train_extr_proccessed, y_train)

print("result")


svm_raw_preddict = svm_proccessed.predict(daisy_test_extr)
print(accuracy_score(y_test, svm_raw_preddict))
svm_processed_preddict = svm_proccessed.predict(daisy_test_extr_proccessed)
print(accuracy_score(y_test, svm_processed_preddict))



print(X_train_processed[289])

fig, ax = plt.subplots(2, 2, figsize=(10,20))
ax = ax.ravel()

ax[0].imshow(X_train[289])
ax[1].imshow(X_train_processed[289])
ax[2].imshow(X_test[0])
ax[3].imshow(X_test_processed[0])
plt.show()






