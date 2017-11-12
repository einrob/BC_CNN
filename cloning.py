# Code from Udacity coursevideo 

import csv 
import cv2
import numpy as np 
import matplotlib.pyplot as plt

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

model = Sequential()



# At the moment we are splitting of validation data 
# from the agmented data which is not the best way to go 

# The validation data youd like to be the best 
# possible measurement of how well your model is doing 
# on what is actually going to see in test - in real life 
#  In "real life" the car is going to see only the 
# center camera data. So in a more better datapipeline 
# we would slicing off the validation set initially before
# augmenting the data and only augment the training data 


# "The Lamda layer allows us to essentially take each feature set
# or image that we pass to e network and do what ever we wan with it"
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))

UseLeNet = False 
NvidiaEndToEnd = True 

if UseLeNet == True: 
	print("Using LeNet")
	# Cropping cropping=((top,bottom),(left,right))
	model.add(Cropping2D(cropping=((70,25), (0,0)))) 
	model.add(Convolution2D(6, 5, 5, activation = 'relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(16, 5, 5, activation = 'relu'))
	model.add(MaxPooling2D())

	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))

if NvidiaEndToEnd == True: 
	print("Using NvidiaEndToEnd")
	# Cropping cropping=((top,bottom),(left,right))
	model.add(Cropping2D(cropping=((70,25), (0,0)))) 
	# subsample == strides 
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation = 'relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
	
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))

	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(50))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Dropout(0.2))
	model.add(Dense(1))




model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 4)

model.save('test_model_3.h5')
