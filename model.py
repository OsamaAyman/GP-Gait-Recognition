import cv2
import numpy as np
import keras


def predect(lista):
    pre=[]
    for img in lista:  # 16 frame
        new_array = cv2.resize(img, (112, 112))
        pre.append(new_array)
    pre = np.array(pre)
    vid=np.expand_dims(pre,axis=0)
    new_model =keras.models.load_model('/train/my_model.h5')
    return np.argmax(new_model.predict(vid)+1)
