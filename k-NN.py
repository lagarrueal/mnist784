#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 01:03:00 2021

@author: cytech
"""
import matplotlib.pyplot as plt
import numpy as np

#opening dataset MNIST directly from scykit
# =============================================================================
# This dataset is made of 70 000 images representing
# handwritten numbers from 0 to 9
# Each image is 28x28 pixels and is composed of a white background and 
# the number written in black
# =============================================================================
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784' , version = 1) 

#main dataset containing all images
print(mnist.data.shape)

#the vector of annotations associated with the dataset (number between 0 and 9)
print(mnist.target.shape)

# =============================================================================
# For the k-NN algorithm, 70000 is to big as a dataset size
# So We're taking a random sample
# =============================================================================

sample = np.random.randint(70000 , size = 5000)
data = mnist.data[sample]
target = mnist.target[sample]

# =============================================================================
# Splitting the sample into train set and test set
# =============================================================================
from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(data , target , train_size = 0.8)

# =============================================================================
# When the sample is split, we can begin to use the model
# =============================================================================
from sklearn import neighbors
# =============================================================================
model = neighbors.KNeighborsClassifier(n_neighbors = 3)
model.fit(xtrain , ytrain)
model.predict([xtest[3]])

#error percentage
print("error percentage on our sample for k = 3 : " , 1 - model.score(xtest, ytest))
# =============================================================================


#method to get error percentage curve by the number of nearest neighbours
errors = []  #creating an empty list
for k in range (2,15): #loop for 2 to 15
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1- knn.fit(xtrain, ytrain).score(xtest, ytest)))

plt.plot(range(2 , 15) ,  errors , 'o-')
plt.show()


# getting the best classifier from the graph
n = int(input("Enter the best classifier : "))
knn = neighbors.KNeighborsClassifier( n )
knn.fit(xtrain, ytrain)

#getting predictions from the test set
predicted = knn.predict(xtest)

#reshaping the date into images
images = xtest.reshape((-1 , 28 , 28))

#getting 12 images randomly
select = np.random.randint(images.shape[0] , size = 16)

#displaying images with associated prediction
fig , ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(4,4,index+1)
    plt.axis('off')
    plt.imshow(images[value] , cmap = plt.cm.gray_r , interpolation = 'nearest')
    plt.title( 'Predicted: {}'.format(predicted[value]) )
    
plt.show()

#getting some false predictions
misclass = (ytest != predicted)
misclass_images = images[misclass,:,:]
misclass_predicted = predicted[misclass]

#getting a sample of those images
select = np.random.randint(misclass_images.shape[0], size=16)

#displayng images and predictions
for index, value in enumerate(select):
    plt.subplot(4,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value], cmap=plt.cm.gray_r , interpolation='nearest')
    plt.title( 'Predicted: {}'.format(misclass_predicted[value]) )
    
plt.show()




