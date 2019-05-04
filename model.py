import csv
import cv2
from scipy import ndimage
import numpy as np

# Read CSV file
lines = []
with open ('../data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Store the images and steering angles for each frames to list       
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data1/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        
        #shift the steering angle depend on the position of the camera
        if i==0: #center
            shift = 0
        elif i==1: #left
            shift = 0.2
        else: # right
            shift = -0.2
        measurement = float(line[3]) + shift
        measurements.append(measurement)
 
# make filp images/steering angle and add them to the dataset
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(-measurement)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(len(X_train))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, Dropout

# Based on NVIDIA's CNN
model = Sequential()
# Normalize the image to -0.5 to 0.5
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Crop the images to remove unnecessary images (Like sky, tree..)
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))

# five CNN layers and two dropout layer
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(0.4))

# Fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) # Steering angle

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
    
    