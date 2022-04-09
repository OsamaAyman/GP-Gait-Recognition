from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

x_train=np.load('/data/X_Train.npy',allow_pickle=True)
y_train=np.load('/data/Y_Train.npy',allow_pickle=True)
x_test=np.load('/data/X_Test.npy',allow_pickle=True)
y_test=np.load('/data/Y_Test.npy',allow_pickle=True)
# print(x_train.shape)


def create_model_sequential():
    model = Sequential()
    input_shape = (16, 112, 112, 3)

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    return model

def create_features_exctractor(C3D_model, layer_name='fc7'):
    extractor = Model(inputs=C3D_model.input,outputs=C3D_model.get_layer(layer_name).output)
    return extractor

def fit_model():
  model = create_model_sequential()

  model.load_weights('C3D_Sport1M_weights_keras_2.2.4.h5')

  model = create_features_exctractor(model)

  new_model = model.output
  new_model = Dense(15, activation="softmax")(new_model)
  model = Model(inputs=model.input, outputs=new_model)
  for layer in model.layers:
      layer.trainable = False

  model.layers[-1].trainable = True
  checkpoint = ModelCheckpoint("train/best_model.hdf5", monitor='loss', verbose=1,
      save_best_only=True, mode='auto', period=1)
  reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                          factor=0.1,
                                          patience=2,
                                          cooldown=2,
                                          min_lr=0.00001,
                                          verbose=1)
  callbacks=reduce_learning_rate
  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  model.fit(x_train,y_train,epochs = 30 , batch_size=32 , shuffle=True, callbacks=[callbacks,checkpoint])
  model.save('train/my_model.h5')
  model.evaluate(x_test,y_test)

fit_model()