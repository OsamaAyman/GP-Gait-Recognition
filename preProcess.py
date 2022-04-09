#this file call two times one for train data and one for test data
from helper import sort
import numpy as np
import os
import cv2
import random
from keras.utils.np_utils import to_categorical

DATADIR = '/data/Train'
CATEGORIES = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"]

training_data = []
training_label = []
testing_data = []
testing_name = []
x_train = []
y_train = []
imglist = []
list_of_pair_test = []
list_of_pair = []
list_of_pair_thief = []
IMG_SIZE = 100
count = 0
Count = 0
dic = {}
b = 0
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    listOfFiles = os.listdir(path)
    for vid in listOfFiles:
        b = b + 1
        count = count + 1
        pathvid = os.path.join(path, vid)
        listOfFrames = os.listdir(pathvid)
        listOfFrames = sort(listOfFrames)
        a = -1
        img_list = []
        for img in listOfFrames:
            a = a + 1
            if a % 16 == 0 and a != 0:
                print('len of imglist' + str(len(img_list)))
                list_of_pair.append([img_list, int(vid[0:3])])
                img_list = []
            if a == 48:
                break
            print(a)

            img_array = cv2.imread(os.path.join(pathvid, img))
            # gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            new_array = cv2.resize(img_array, (112, 112))
            # new_array=cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
            img_list.append(new_array)

        print('len of list' + str(len(list_of_pair)))
        ################################################################################
    # print("5lsna")
    # print(len(training_data))
print(len(list_of_pair))  # 135
np.save('/data/newx.npy', list_of_pair)



lista_train = []
training_data1_train = []
training_label1_train = []
for i in range(len(list_of_pair)):
    lista_train.append((list_of_pair[i][0], list_of_pair[i][1] - 1))
random.shuffle(lista_train)
# print(lista)
a = 0
for i, j in lista_train:
    training_data1_train.append(i)
    training_label1_train.append(j)
    a = a + 1

training_data1_train = np.array(training_data1_train)
training_label1_train = np.array(training_label1_train)

nb_classes = 15
print(training_label1_train)

training_label1_train = to_categorical(training_label1_train, nb_classes)

x_train = training_data1_train
y_train = training_label1_train
np.save('/data/train.npy',x_train)
np.save('/data/test.npy',y_train)

print(x_train.shape)
print(y_train.shape)