# Code from Udacity coursevideo 

import argparse
import csv 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import load_model


lines = []

print("Loading file paths ... ")

with open('./data/driving_log.csv') as csvfile: 
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)


#print (lines)

print("File paths loaded ... ")


images = []
measurements = []


for line in lines: 
	# Loading three images field 0, 1, 2 
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		#plt.figure(figsize=(5,5))
		#plt.imshow(image)
	
	measurement = float(line[3])
	# If car is weaving side to side it may help to reduce 
	# this factor 
	correction = 0.2 
	# Correcting steering agles 
	measurements.append(measurement) # Center image 
	measurements.append(measurement + correction) # Left image 
	measurements.append(measurement - correction) # Right image 

# steering angles are normalized for us 


# Generate a Histogram with 43 bins 

# the histogram of the data
plt.hist(measurements, 42, normed=0, facecolor='green', alpha=0.75)

plt.xlabel('Samples of steering angle')
plt.ylabel('Count of steering angle')
plt.axis([-1, 1, 0, 1000])
plt.grid(True)

plt.show()



print(len(images))
print(len(measurements))

augmented_images = []
augmented_measurements = [] 

for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image,1)
	flipped_measurement = measurement * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print (X_train.shape)


# KREAS 
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Dropout  
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D 



parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'model',
    type=str,
    help='Path to model h5 file. Model should be on the same path.'
)

args = parser.parse_args()
print ("Loading model file: ", args.model)

## Load and retrain model 

model = None

model = load_model(args.model)


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 4)

model.save('test_model_5_on_413.h5')
