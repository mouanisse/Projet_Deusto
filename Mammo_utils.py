import os
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.applications import VGG16





class Mammographie:

    def __init__(self, INCAN_size, numpy_database_path, height, width):
        self.database_size = INCAN_size
        self.database_path = None
        self.height = height
        self.width = width
        self.train_data = np.load(numpy_database_path+'/train_data_('+str(self.height)+','+str(self.width)+')_2112.npy')
        self.val_data = np.load(numpy_database_path+'/val_data_('+str(self.height)+','+str(self.width)+')_2112.npy')
        self.test_data = np.load(numpy_database_path+'/test_data_('+str(self.height)+','+str(self.width)+')_2112.npy')
        self.train_labels = np.load(numpy_database_path+'/train_labels_('+str(self.height)+','+str(self.width)+')_2112.npy')
        self.val_labels = np.load(numpy_database_path+'/val_labels_('+str(self.height)+','+str(self.width)+')_2112.npy')
        self.test_labels = np.load(numpy_database_path+'/test_labels_('+str(self.height)+','+str(self.width)+')_2112.npy')
        self.input_shape = (height, width, 1)


    def database_preprocessing(self):
        train_data = []
        val_data = []
        test_data = []
        train_labels = []
        val_labels = []
        test_labels = []


        for root, dirs, files in os.walk(self.database_path + '/', topdown=False):
            # Loop through files

            train_iteration = math.floor(0.8 * len(files))
            val_iteration = math.floor(0.1 * len(files))
            test_iteration = len(files) - (math.floor(0.8 * len(files)) + math.floor(0.1 * len(files)))


            if root[-1:] == '0':  # normal images

                for i in range(train_iteration):

                    if files[i] != '.DS_Store' and files[i] != '.DS_Stoe**':
                        train_labels.append(0)
                        train_data.append(np.array(Image.open(root + '/' + files[i])))

                for j in range(train_iteration, train_iteration + val_iteration):

                    if files[j] != '.DS_Store' and files[j] != '.DS_Stoe**':
                        val_labels.append(0)
                        val_data.append(np.array(Image.open(root + '/' + files[j])))

                for k in range(train_iteration + val_iteration,
                                train_iteration + val_iteration + test_iteration):

                    if files[k] != '.DS_Store' and files[k] != '.DS_Stoe**':
                        test_labels.append(0)
                        test_data.append(np.array(Image.open(root + '/' + files[k])))



            elif root[-1:] == '8':  # anormal images

                for i in range(train_iteration):

                    if files[i] != '.DS_Store' and files[i] != '.DS_Stoe**':
                        train_labels.append(1)
                        train_data.append(np.array(Image.open(root + '/' + files[i])))

                for j in range(train_iteration, train_iteration + val_iteration):

                    if files[j] != '.DS_Store' and files[j] != '.DS_Stoe**':
                        val_labels.append(1)
                        val_data.append(np.array(Image.open(root + '/' + files[j])))

                for k in range(train_iteration + val_iteration,
                               train_iteration + val_iteration + test_iteration):

                    if files[k] != '.DS_Store' and files[k] != '.DS_Stoe**':
                        test_labels.append(1)
                        test_data.append(np.array(Image.open(root + '/' + files[k])))


        # Convert lists into arrays
        train_labels = np.asarray(train_labels)
        val_labels = np.asarray(val_labels)
        test_labels = np.asarray(test_labels)
        train_data = np.asarray(train_data)
        val_data = np.asarray(val_data)
        test_data = np.asarray(test_data)


        # Shuffle the data and the labels simultanousely
        train_data, train_labels = shuffle(train_data, train_labels)
        val_data, val_labels = shuffle(val_data, val_labels)
        test_data, test_labels = shuffle(test_data, test_labels)



        # To get shape (129, 129, 1)
        train_data = np.expand_dims(train_data, 3)
        val_data = np.expand_dims(val_data, 3)
        test_data = np.expand_dims(test_data, 3)
        train_labels = np.expand_dims(train_labels, -1)
        val_labels = np.expand_dims(val_labels, -1)
        test_labels = np.expand_dims(test_labels, -1)


        # We fill the variables defined by 'None', with their respective values
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels



        print(train_labels.max())
        print(train_labels.min())
        print(train_data.max())
        print(train_data.min())

        print('*********************************')

        print(train_data.shape)
        print(val_data.shape)
        print(test_data.shape)
        print(train_labels.shape)
        print(val_labels.shape)
        print(test_labels.shape)



        np.save('Projet_Deusto/Numpy_dataset/train_data_('+str(self.height)+','+str(self.width)+')_2112.npy', train_data)
        np.save('Projet_Deusto/Numpy_dataset/val_data_(' + str(self.height) + ',' + str(self.width) + ')_2112.npy', val_data)
        np.save('Projet_Deusto/Numpy_dataset/test_data_('+str(self.height)+','+str(self.width)+')_2112.npy', test_data)


        np.save('Projet_Deusto/Numpy_dataset/train_labels_('+str(self.height)+','+str(self.width)+')_2112.npy', train_labels)
        np.save('Projet_Deusto/Numpy_dataset/val_labels_(' + str(self.height) + ',' + str(self.width) + ')_2112.npy', val_labels)
        np.save('Projet_Deusto/Numpy_dataset/test_labels_('+str(self.height)+','+str(self.width)+')_2112.npy', test_labels)



    def model(self):
    

        # 4(2D_CONV_LAYERS + Batch_Norm + 2DMaxPooling) + 2FULLY_CONNECTED + 1SOFTMAX

        model = keras.Sequential()

        model.add(keras.layers.Conv2D(40, (5, 1), activation='relu', input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image

        model.add(keras.layers.Conv2D(80, (4, 4), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image

        model.add(keras.layers.Conv2D(160, (3, 3), activation='relu'))#, activity_regularizer=keras.regularizers.l1(0.001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))
        
        model.add(keras.layers.Conv2D(320, (3, 3), activation='relu'))#, activity_regularizer=keras.regularizers.l1(0.001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        

        return model
 

    def train(self):

        # Define epoch and batch_size
        epochs = 30
        batch_size = 100

        # First, we need to prepare our training and testing data, and pre-process it
        print('The database is already pre-processed!!')

        
        # Then, create our neural network model
        print('Starting building the model ...')
        model = self.model()
        print('Model built !!')
        

        # We perform DataAugmentation, since we have only 718 samples in total, we use Keras ImageDataGenerator object
        
        aug = keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])



        # We compile our model using adam optimizer and binary_crossentropy
        print('Starting compiling the model ...')
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        print('Compiling done !!')
        
        
        print('Starting the training...')

        # We train our model using the batch data generated by flow() function
        
        # Visualizing the results using Tensorboard as a backend
        
        TensorBoard = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)
        
        # Performing Early stopping to avoid overfitting
        
        EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=6)
        
        
        # Perform checkpoint
        filepath = "best_weights.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [TensorBoard, checkpoint]
        
        
        # model.fit_generator(aug.flow(self.train_data, self.train_labels, batch_size=batch_size)
                      #  ,steps_per_epoch=len(self.train_data)//batch_size, epochs=epochs
                     #  , validation_data=(self.val_data, self.val_labels), callbacks=[CallBack])
                
        
                
        model.fit(self.train_data, self.train_labels, batch_size=batch_size, epochs=epochs,
             verbose=1, validation_data=(self.val_data, self.val_labels), callbacks=callbacks_list)
        
            
        print('Training done !!')
        
        # load weights
        model.load_weights("best_weights.hdf5")
        
        
        
        # We compile the model used for predictions
        print('Starting compiling the model used for predictions...')
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        print("Created model and loaded weights from file !!")
        
        

        # We test our model using the test dataset
        score = model.evaluate(self.test_data, self.test_labels, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        
        
        # verbose shows you the training progress for each epoch, 0 is silent, 1 will show an animated progress bar "[=======]"
        # , and 2 will just mention the number of Epoch "Epoch 1/10"
        # ATTENTION: predict function returned an array of probabilities of the second class, and not the first one.
        # score = model.predict(self.test_data)
        # predicted_labels = (score > 0.5).astype(np.int)
        predicted_labels = model.predict_classes(self.test_data)
        actual_labels = self.test_labels.astype(np.int)
        
        
        #Compute the confusion matrix
        matrix = confusion_matrix(actual_labels, predicted_labels)
        print(matrix)
        
        
        #Save the model in h5 format
        model.save('mammo.h5')
        

        


mammo = Mammographie(2112, '/content/drive/My Drive/Colab Notebooks/Projet_Deusto/Numpy_dataset', 129, 129)
mammo.train()








































