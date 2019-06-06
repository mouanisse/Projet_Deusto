import os
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras






class Mammographie:

    def __init__(self, INCAN_size, numpy_database_path, height, width):
        self.database_size = INCAN_size
        self.database_path = None
        self.height = height
        self.width = width
        self.train_data = np.load(numpy_database_path+'/train_data_(129,129).npy')
        self.val_data = np.load(numpy_database_path+'/val_data_(129,129).npy')
        self.test_data = np.load(numpy_database_path+'/test_data_(129,129).npy')
        self.train_labels = np.load(numpy_database_path+'/train_labels_(129,129).npy')
        self.val_labels = np.load(numpy_database_path+'/val_labels_(129,129).npy')
        self.test_labels = np.load(numpy_database_path+'/test_labels_(129,129).npy')
        self.input_shape = (height, width, 1)


    def database_preprocessing(self):
        tumor = np.ones((self.database_size,1))
        npy_dataset = np.empty([self.database_size, self.height, self.width])
        for root, dirs, files in os.walk(self.database_path + '/', topdown=False):
            # Loop through files
            if root[-1:] == '0':  # normal images
                k = 0
                for f in files:
                    if f != '.DS_Store' and f != '.DS_Stoe**':
                        tumor[k] = 0
                        npy_dataset[k][:][:][:] = np.array(Image.open(root + '/' + f))
                        k = k + 1


            elif root[-1:] == '8':  # anormal images
                k = 0
                for f in files:
                    if f != '.DS_Store' and f != '.DS_Stoe**':
                        npy_dataset[k][:][:][:] = np.array(Image.open(root + '/' + f))
                        k = k + 1

        # Transform the labels to use categorical_crossentropy (more than 2 classes) as a loss function
        labels = tumor

        # Split into training and testing sets
        train_data, test_val_data, train_labels, test_val_labels = train_test_split(npy_dataset, labels, test_size=0.2,
                                                                                    random_state=42)
        test_data, val_data, test_labels, val_labels = train_test_split(test_val_data, test_val_labels, test_size=0.5,
                                                                                    random_state=42)

        # To get shape (129, 129, 1)
        train_data = np.expand_dims(train_data, 3)
        val_data = np.expand_dims(val_data, 3)
        test_data = np.expand_dims(test_data, 3)

        # We fill the variables defined by 'None', with their respective values
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.input_shape = (129, 129, 1)
        
        print(labels.max())
        print(labels.min())
        print(train_data[0].max())
        print(train_data[0].min())



    def model(self):

        # 4(2D_CONV_LAYERS + Batch_Norm + 2DMaxPooling) + 2FULLY_CONNECTED + 1SOFTMAX

        model = keras.Sequential()


        model.add(keras.layers.Conv2D(20, (5, 1), activation='relu', input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image

        model.add(keras.layers.Conv2D(40, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image

        model.add(keras.layers.Conv2D(80, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))

        model.add(keras.layers.Conv2D(160, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        return model


    def train(self):

        # Define epoch and batch_size
        epochs = 50
        batch_size = 30

        # First, we need to prepare our training and testing data, and pre-process it
        print('The database is already pre-processed ...')

        # Then, create our neural network model
        print('Starting building the model ...')
        model = self.model()
        print('Model built !!')

        # We perform DataAugmentation, since we have only 718 samples in total, we use Keras ImageDataGenerator object
        aug = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                 horizontal_flip=True, fill_mode="nearest")



        # We compile our model using adam optimizer and binary_crossentropy
        print('Starting compiling ...')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Compiling done !!')


        # We train our model using the batch data generated by flow() function
        print('Starting the training...')
        #model.fit_generator(aug.flow(self.train_data, self.train_labels, batch_size=batch_size)
                               # ,steps_per_epoch=len(self.train_data)//batch_size, epochs=epochs
                           # ,validation_data=(self.val_data, self.val_labels))
                
        model.fit(self.train_data, self.train_labels, batch_size=30, epochs=20,
                  validation_data=(self.val_data, self.val_labels))
        print('Training done !!')


        # We test our model using the test dataset
        score = model.evaluate(self.test_data, self.test_labels, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])








mammo = Mammographie(718, '/content/drive/My Drive/Colab Notebooks/Projet_Deusto/Numpy_dataset', 129, 129)
mammo.train()








































