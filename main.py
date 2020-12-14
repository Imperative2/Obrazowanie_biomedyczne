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
from skimage.feature import local_binary_pattern
from skimage.feature import CENSURE
from sklearn.preprocessing import Binarizer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True)
from PIL import Image
from skimage.feature import (corner_harris, corner_peaks, BRIEF, match_descriptors)
import timeit
from threading import Thread
from time import sleep

def calculate_daisy(images, daisy_list):
    for img in images:
        #print("daisy: ",len(daisy_train))
        #DAISY
        descs = daisy(img, step=int(img.shape[0] / 2), radius=int(img.shape[1] / 5), rings=2, histograms=8,
                             orientations=8, visualize=False)
        daisy_list.append(np.asarray(descs).reshape(-1))


def calculate_hog(images, hog_list):
    for img in images:
        #print("hog: ", len(hog_list))
        #HOG
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(int(img.shape[0]/10), int(img.shape[1]/10)),
                            cells_per_block=(1,1), visualize=True)
        hog_list.append(np.asarray(fd).reshape(-1))

def calculate_lbp(images, lbp_list):
    for img in images:
        #print("lbp: ", len(lbp_list))
        #LBP
        out  = local_binary_pattern(img, P=img.shape[0]/30*4, R=img.shape[0]/30)
        lbp_list.append(np.asarray(out).reshape(-1))

def calculate_orb(images, orb_list):
    for img in images:
        #ORB
        orbDetector = ORB(n_keypoints=50, fast_threshold=0.01)
        orbDetector.detect_and_extract(img)
        orb_features = orbDetector.descriptors
        orb_list.append(np.asarray(orb_features).reshape(-1))

def calculate_brief(images, brief_list):
    for img in images:
        print("brief: ", len(brief_list))
        #BRIEF
        keypoints = corner_harris(img)
        print(keypoints)

        # censure = CENSURE()
        # censure.detect(img)
        #print(len(censure.keypoints))
        extractor = BRIEF()
        extractor.extract(img, keypoints)
        keypoints1 = keypoints[extractor.mask]
        brief_list.append(np.asarray(extractor.descriptors).reshape(-1))




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


#ładowanie zdjęć do list
print("loading pictures into memory")
for path in train_images_normal_path:
    im = np.asarray(Image.open('./archive/TrainImages/Normal/'+path))
    if im.ndim == 3:
        im = np.sum(im, axis=2).astype(np.float)
        min = im.min(axis=0).min()
        max = im.max(axis=0).max()
        im = ((im - min) / max ) * 255
        im = im.astype(np.uint8)
    X_train.append(im)
    y_train.append(labels[0])
for path in train_images_covid_path:
    im = np.asarray(Image.open('./archive/TrainImages/COVID-19/'+path))
    if im.ndim == 3:
        im = np.sum(im, axis=2).astype(np.float)
        min = im.min(axis=0).min()
        max = im.max(axis=0).max()
        im = ((im - min) / max ) * 255
        im = im.astype(np.uint8)
    X_train.append(im)
    y_train.append(labels[1])
for path in test_images_normal_path:
    im = np.asarray(Image.open('./archive/TestImages/Normal/'+path))
    if im.ndim == 3:
        im = np.sum(im, axis=2).astype(np.float)
        min = im.min(axis=0).min()
        max = im.max(axis=0).max()
        im = ((im - min) / max ) * 255
        im = im.astype(np.uint8)
    X_test.append(im)
    y_test.append(labels[0])
for path in test_images_covid_path:
    im = np.asarray(Image.open("./archive/TestImages/COVID-19/"+path))
    if im.ndim == 3:
        im = np.sum(im, axis=2).astype(np.float)
        min = im.min(axis=0).min()
        max = im.max(axis=0).max()
        im = ((im - min) / max ) * 255
        im = im.astype(np.uint8)
    X_test.append(im)
    y_test.append(labels[1])



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
    #plt.imsave(str(i)+".jpg", scaled, cmap="gray")

    X_train_all[0][i] = (scaled * 255).astype("uint8")

    sharpness = unsharp_mask(scaled, radius=1, amount=1)
    X_train_all[1][i] = (sharpness * 255).astype("uint8")

    med = median(scaled, square(3))
    X_train_all[2][i] = (med * 255).astype("uint8")


for i in range(X_test_all.shape[1]):
    print(i)
    scaled = resize(X_test[i], output_shape=(min_height, min_width))
    #plt.imsave("test_"+str(i)+".jpg", scaled, cmap="gray")

    X_test_all[0][i] = (scaled * 255).astype("uint8")

    sharpness = unsharp_mask(scaled, radius=1, amount=1)
    X_test_all[1][i] = (sharpness * 255).astype("uint8")

    med = median(scaled, square(3))
    X_test_all[2][i] = (med * 255).astype("uint8")



# eksperyment 1 sprawdzający wpływ przetwarzania na jakosc klasyfikacji, eksperyment 2 badajacy wpływ deskryptorów na
# jakość klasyfikacji
exp_1_2_results = np.zeros(shape=(3,5))

for i in range(X_train_all.shape[0]):
    X_train_processed = X_train_all[i]
    X_test_processed = X_test_all[i]

    daisy_train = []
    thread_daisy = Thread(target=calculate_daisy, args=(X_train_processed, daisy_train))
    thread_daisy.start()

    hog_train=[]
    thread_hog = Thread(target=calculate_hog, args=(X_train_processed, hog_train))
    thread_hog.start()

    lbp_train=[]
    thread_lbp = Thread(target=calculate_lbp, args=(X_train_processed, lbp_train))
    thread_lbp.start()

    orb_train=[]
    thread_orb = Thread(target=calculate_orb, args=(X_train_processed, orb_train))
    thread_orb.start()

    # brief_train=[]
    # thread_brief = Thread(target=calculate_brief, args=(X_train_processed, brief_train))
    # thread_brief.start()

    thread_daisy.join()
    thread_hog.join()
    thread_lbp.join()
    thread_orb.join()
    # thread_brief.join()

    daisy_test = []
    thread_daisy = Thread(target=calculate_daisy, args=(X_test_processed, daisy_test))
    thread_daisy.start()

    hog_test = []
    thread_hog = Thread(target=calculate_hog, args=(X_test_processed, hog_test))
    thread_hog.start()

    lbp_test = []
    thread_lbp = Thread(target=calculate_lbp, args=(X_test_processed, lbp_test))
    thread_lbp.start()

    orb_test = []
    thread_orb = Thread(target=calculate_orb, args=(X_test_processed, orb_test))
    thread_orb.start()

    # brief_test=[]
    # thread_brief = Thread(target=calculate_brief, args=(X_test_processed, brief_test))
    # thread_brief.start()

    thread_daisy.join()
    thread_hog.join()
    thread_lbp.join()
    thread_orb.join()
    # thread_brief.join()

    #training
    svm_daisy = LinearSVC(max_iter=10000)
    svm_hog = LinearSVC(max_iter=10000)
    svm_ORB = LinearSVC(max_iter=10000)
    svm_LBP = LinearSVC(max_iter=10000)
    svm_brief = LinearSVC(max_iter=10000)

    print("training daisy")
    start = timeit.timeit()
    svm_daisy.fit(daisy_train,y_train)
    daisy_predict = svm_daisy.predict(daisy_test)
    end = timeit.timeit()
    print(end-start)

    print("daisy prediction: ", accuracy_score(y_test, daisy_predict))
    exp_1_2_results[i][0] =  accuracy_score(y_test, daisy_predict)

    print("training HOG")
    start = timeit.timeit()
    svm_hog.fit(hog_train,y_train)
    hog_predict = svm_hog.predict(hog_test)
    end = timeit.timeit()
    print(end-start)
    print("hog prediction: ", accuracy_score(y_test, hog_predict))
    exp_1_2_results[i][1] =  accuracy_score(y_test, hog_predict)

    print("training ORB")
    start = timeit.timeit()
    svm_ORB.fit(orb_train,y_train)
    orb_predict = svm_ORB.predict(orb_test)
    end = timeit.timeit()
    print(end-start)
    print("orb prediction: ", accuracy_score(y_test, orb_predict))
    exp_1_2_results[i][2] =  accuracy_score(y_test, orb_predict)

    print("training LBP")
    start = timeit.timeit()
    svm_LBP.fit(lbp_train,y_train)
    lbp_predict = svm_LBP.predict(lbp_test)
    end = timeit.timeit()
    print(end-start)
    print("LBP prediction: ", accuracy_score(y_test, lbp_predict))
    exp_1_2_results[i][3] =  accuracy_score(y_test, lbp_predict)

    # print("training BRIEF")
    # start = timeit.timeit()
    # svm_brief.fit(brief_train,y_train)
    # brief_predict = svm_LBP.predict(brief_test)
    # end = timeit.timeit()
    # print(end-start)
    # print("BRIEF prediction: ", accuracy_score(y_test, brief_predict))
    # exp_1_2_results[i][3] =  accuracy_score(y_test, brief_predict)


#eksperyment 3 badanie argumentów deskryptora

exp_3_results = np.zeros(shape=(3*3*3*3))

orientations = [4,9,16]
pixels_per_cell = [(4,4),(8,8),(16,16)]
cells_per_block = [(1,1),(3,3),(9,9)]
block_norms = ["L1","L2","L2-Hys"]

i = 0

for orientation in orientations:
    for pixel_per_cell in pixels_per_cell:
        for cell_per_block in cells_per_block:
            for block_norm in block_norms:
                X_train_processed = X_train_all[1]
                X_test_processed = X_test_all[1]
                hog_train = []
                hog_test = []

                for img in X_train_processed:
                    fd = hog(img, orientations=orientation,
                                        pixels_per_cell=pixel_per_cell,
                                        cells_per_block=cells_per_block,block_norm=block_norm)
                    hog_train.append(np.asarray(fd).reshape(-1))
                for img in X_test_processed:
                    fd = hog(img, orientations=orientation,
                                        pixels_per_cell=pixel_per_cell,
                                        cells_per_block=cells_per_block,block_norm=block_norm)
                    hog_test.append(np.asarray(fd).reshape(-1))

                svm_hog = LinearSVC(max_iter=10000)
                print("training HOG")
                start = timeit.timeit()
                svm_hog.fit(hog_train, y_train)
                hog_predict = svm_hog.predict(hog_test)
                end = timeit.timeit()
                print(end - start)
                print("hog prediction: ", accuracy_score(y_test, hog_predict))
                exp_3_results[i] = accuracy_score(y_test, hog_predict)
                i+=1

exp_3_results.reshape(shape=(3,3,3,3))
