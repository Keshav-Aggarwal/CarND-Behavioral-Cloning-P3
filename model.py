import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.layers import Dense, Flatten, Activation, Lambda, Cropping2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
get_ipython().magic('matplotlib inline')

import csv
import matplotlib.pyplot as plt
import os

print('Loading Data ...')
images = []
measurements = []
#LOAD ALL THE IMAGES FROM CSV
def load_all_csv(paths):
    lines = []
    fol_num = -2
    for path in paths:
        lines_temp = []
        fol_num = fol_num + 2
        print('Getting data for :', path)
        with open(path) as file:
            reader = csv.reader(file)
            for line in reader:
                lines_temp.append(line)
        lines_temp = lines_temp[1:]
        print('Number of Sample images are ', len(lines_temp))
        
        lines_temp_measurement = np.asarray(lines_temp)[:,3:4]
        lines_temp_measurement = np.asarray(lines_temp_measurement, dtype=float)
        
        #GET THE INDEX OF ALL ZERO AND NON ZERO ELEMENTS
        zero_indices = np.where(lines_temp_measurement == 0.0)
        non_zero_indices = np.nonzero(lines_temp_measurement)
        
        #GET THE 25% RANDOM ZERO SAMPLES FROM THE ZERO STEERING ANGLE DATA
        result_indices_zero = np.random.choice(zero_indices[0], int(len(zero_indices[0])/4))
        result_indices = np.concatenate((non_zero_indices[0], result_indices_zero))
        
        #LOAD THE IMAGE PATH AND STEERING ANGLE IN IMAGES AND MEASUREMENTS VARIABLE
        for line in result_indices:
            
            for i in range(3):
                current_path = lines_temp[line][i].replace('\\','/')

                filename = os.path.split(current_path)[-1]
                filepath = os.path.join(str(fol_num)+'/IMG/', filename) #+ '.jpg'
                images.append(filepath)
                measurement = float(lines_temp[line][3])
                if(i == 2):
                    measurement = float(lines_temp[line][3]) + (-0.2)
                if(i == 1):
                    measurement = float(lines_temp[line][3]) + 0.2
                    
                measurements.append(measurement)
    return images, measurements
#PATH OF THE CSV FILES
images, measurements = load_all_csv(["0/driving_log.csv","1/driving_log.csv","4/driving_log.csv","6/driving_log.csv","8/driving_log.csv","10/driving_log.csv","12/driving_log.csv","14/driving_log.csv","16/driving_log.csv"]) 
print("Data with all Cameras is ",len(images))
print("Data Loaded")

#Removing 0 and adjacent data by 75%.
itemCount = len(measurements)
print("Total data ", itemCount)
print("Total non zero values ", np.count_nonzero(measurements))
print("Total zero values ", itemCount - np.count_nonzero(measurements))

#READ AND RETURN THE IMAGE FROM THE PATH
def readImg(path):
    return plt.imread(path)

#GENERATOR WILL LOAD ALL THE IMAGES FROM IMAGE PATH AND THEN IT WILL LOAD THE IMAGES FROM THAT PATH AND RETURN IT TO THE MODEL FOR TRAINING
def generator(images_data, measurement_data, batch_size=32):
    num_samples = len(images_data)
    while 1:
        shuffle(images_data, measurement_data)
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_images_path = []
            batch_images_path = np.array(images_data[offset: offset + batch_size])
            batch_measurements = np.array(measurement_data[offset: offset + batch_size])
            
            for img in batch_images_path:
                batch_images.append(readImg(img))
            
            flipped_img = np.concatenate((batch_images,np.fliplr(batch_images)))
            flipped_measurement = np.concatenate((batch_measurements, (batch_measurements*(-1))))
            
            yield shuffle(flipped_img, flipped_measurement)

X_train, X_valid,y_train, y_valid = train_test_split(images, measurements, test_size = 0.2)

train_generator = generator(X_train, y_train, batch_size = 64)
valid_generator = generator(X_valid, y_valid, batch_size = 32)

#DECLARE THE MODEL
print("Declaring Model")
mo  del = Sequential()
model.add(Lambda(lambda x: (x / 127.5)-1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((58,24),(0,0))))

model.add(Convolution2D(24,5,5, subsample = (2,2), name='Conv1', activation='relu'))
model.add(Convolution2D(36,5,5, subsample = (2,2), name='Conv2', activation='relu'))
model.add(Convolution2D(48,5,5, subsample = (2,2), name='Conv3', activation='relu'))
model.add(Dropout(0.2, name='Drop1'))
model.add(Convolution2D(64,3,3, subsample = (2,2), name='Conv4', activation='relu'))
model.add(Convolution2D(64,3,3, subsample = (1,1), name='Conv5', activation='relu'))

model.add(Flatten(name='Flat1'))
model.add(Dropout(0.2, name='Drop2'))
model.add(Dense(100, name='Dense1', activation='elu'))          
model.add(Dense(50, name='Dense2', activation='elu'))
model.add(Dropout(0.2, name='Drop3'))
model.add(Dense(10, name='Dense3', activation='elu'))
model.add(Dense(1, name='Output_Layer'))

print("Model Declared")
model.summary()

model.compile(loss='mse', optimizer= Adam(lr = 0.0001))
model.fit_generator(train_generator, samples_per_epoch=len(X_train),validation_data= valid_generator, nb_val_samples=len(X_valid), nb_epoch=3)
#SAVE THE MODEL
model.save('model.h5')