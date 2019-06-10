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





class Mammographie:

    def __init__(self, INCAN_size, numpy_database_path, height, width):
        self.database_size = INCAN_size
        self.database_path = None
        self.height = height
        self.width = width
        self.train_data = np.load(numpy_database_path+'/train_data_('+str(self.height)+','+str(self.width)+')_mini.npy')
        self.val_data = np.load(numpy_database_path+'/val_data_('+str(self.height)+','+str(self.width)+')_mini.npy')
        self.test_data = np.load(numpy_database_path+'/test_data_('+str(self.height)+','+str(self.width)+')_mini.npy')
        self.train_labels = np.load(numpy_database_path+'/train_labels_('+str(self.height)+','+str(self.width)+')_mini.npy')
        self.val_labels = np.load(numpy_database_path+'/val_labels_('+str(self.height)+','+str(self.width)+')_mini.npy')
        self.test_labels = np.load(numpy_database_path+'/test_labels_('+str(self.height)+','+str(self.width)+')_mini.npy')
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
        
        print(labels.max())
        print(labels.min())
        print(train_data[0].max())
        print(train_data[0].min())



    def model(self):

        # 4(2D_CONV_LAYERS + Batch_Norm + 2DMaxPooling) + 2FULLY_CONNECTED + 1SOFTMAX

        model = keras.Sequential()


        model.add(keras.layers.Conv2D(40, (5, 1), activation='relu', input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image

        model.add(keras.layers.Conv2D(80, (4, 4), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))  # divise par 4 les dimensions de l'image

        model.add(keras.layers.Conv2D(160, (3, 3), activation='relu', activity_regularizer=keras.regularizers.l1(0.001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))
        
        model.add(keras.layers.Conv2D(320, (3, 3), activation='relu', activity_regularizer=keras.regularizers.l1(0.001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(2, 2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        

        return model
 

    def train(self):

        # Define epoch and batch_size
        epochs = 150
        batch_size = 30

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
        print('Starting compiling ...')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Compiling done !!')
        
        
        print('Starting the training...')

        # We train our model using the batch data generated by flow() function
        
        # Visualizing the results using Tensorboard as a backend
        
        CallBack = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)
        
        
        model.fit_generator(aug.flow(self.train_data, self.train_labels, batch_size=batch_size)
                             ,steps_per_epoch=len(self.train_data)//batch_size, epochs=epochs
                          , validation_data=(self.val_data, self.val_labels), callbacks=[CallBack])
                
        
                
        #model.fit(self.train_data, self.train_labels,
                 #  batch_size=batch_size, epochs=epochs,
               #   verbose=1, validation_data=(self.val_data, self.val_labels),
                      #   callbacks=[CallBack])
        
            
        print('Training done !!')


        # We test our model using the test dataset
        score = model.evaluate(self.test_data, self.test_labels, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        # verbose shows you the training progress for each epoch, 0 is silent, 1 will show an animated progress bar "[=======]"
        # , and 2 will just mention the number of Epoch "Epoch 1/10"
        # ATTENTION: predict function returned an array of probabilities of the second class, and not the first one.
        # score = model.predict(self.test_data)
        # predicted_labels = (score<0.5).astype(np.int)
        predicted_labels = model.predict_classes(self.test_data)
        actual_labels = self.test_labels.astype(np.int)
        print('Actual labels: ', actual_labels)
        print('Predicted labels: ', predicted_labels)
        
        #Compute the confusion matrix
        matrix = confusion_matrix(actual_labels, predicted_labels)
        print(matrix)
        
        #history_dict = history.history
        #loss_values = history_dict['loss']
        #val_loss_values = history_dict['val_loss']
        #acc = history_dict['acc']
        #val_acc = history_dict['val_acc']

        #Epochs = range(1, len(loss_values)+1)

        #plt.plot(Epochs, loss_values, 'r', label='Training loss')
        #plt.plot(Epochs, val_loss_values, 'b', label='Validation loss')
        #plt.title('Training and Validation loss')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()

        #plt.show()


        #plt.plot(Epochs, acc, 'r', label='Training acc')
        #plt.plot(Epochs, val_acc, 'b', label='Validation acc')
        #plt.title('Training and Validation accuracy')
        #plt.xlabel('Epochs')
        #plt.ylabel('Accuracy')
        #plt.legend()

        #plt.show()
        
        



mammo = Mammographie(380, '/content/drive/My Drive/Colab Notebooks/Projet_Deusto/Numpy_dataset', 129, 129)
mammo.train()








































