#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 01:03:00 2021

@author: cytech
"""
import matplotlib.pyplot as plt
import numpy as np

#ouverture du jeu de données MNIST directement depuis scykit
# =============================================================================
# Ce jeu de données est constitué de 70 000 images représentant 
# l'écriture manuscrite des chiffres de 0 à 9
# Chaque image fait 28x28 pixels et est composée d'un fond blanc et du
# chiffre écrit en noir
# =============================================================================
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784' , version = 1) 

#le dataset principal qui contient toutes les images
print(mnist.data.shape)

#le vecteur d'annotaions associés au dataset (nombre entre 0 et 9)
print(mnist.target.shape)

# =============================================================================
# Pour l'algorithme des k-NN, 70000 est deja trop comme nombre de données
# On va donc faire un échantillon aléatoire
# =============================================================================

sample = np.random.randint(70000 , size = 5000)
data = mnist.data[sample]
target = mnist.target[sample]

# =============================================================================
# L'échantillon étant créé, il faut maintenant séparer les train_data et les
# test_data
# =============================================================================
from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(data , target , train_size = 0.8)

# =============================================================================
# Une fois les données séparées, on peut commencer à utiliser le modèle
# =============================================================================
from sklearn import neighbors
# =============================================================================
model = neighbors.KNeighborsClassifier(n_neighbors = 3)
model.fit(xtrain , ytrain)
model.predict([xtest[3]])

#pourcentage d'erreur
print("pourcentage d'erreur sur notre échantillon pour k = 3 : " , 1 - model.score(xtest, ytest))
# =============================================================================


#méthode pour avoir la courbe de pourcentage d'erreur en fonction du nombre de plus proche voisins
errors = []  #création d'une list vide
for k in range (2,15): #création d'une boucle allant de 2 à 15
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1- knn.fit(xtrain, ytrain).score(xtest, ytest)))

plt.plot(range(2 , 15) ,  errors , 'o-')
plt.show()


#à partir du graphique précédant on récupère le classifieur le plus performant
n = int(input("Entrez le classifieur le plus performant : "))
knn = neighbors.KNeighborsClassifier( n )
knn.fit(xtrain, ytrain)

#on récupère les prédictions sur les données test
predicted = knn.predict(xtest)

#on redimensionne les données sous forme d'images
images = xtest.reshape((-1 , 28 , 28))

#on séléctione 12 images au hasard
select = np.random.randint(images.shape[0] , size = 16)

#on affiche les images avec la prédiction associée
fig , ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(4,4,index+1)
    plt.axis('off')
    plt.imshow(images[value] , cmap = plt.cm.gray_r , interpolation = 'nearest')
    plt.title( 'Predicted: {}'.format(predicted[value]) )
    
plt.show()

#on peut également récupérer quelques prédictions éronées pour mieux comprendre l'algorithme
misclass = (ytest != predicted)
misclass_images = images[misclass,:,:]
misclass_predicted = predicted[misclass]

#on séléctionne un échantillon de ces images
select = np.random.randint(misclass_images.shape[0], size=16)

#on affiche les images et les prédictions
for index, value in enumerate(select):
    plt.subplot(4,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value], cmap=plt.cm.gray_r , interpolation='nearest')
    plt.title( 'Predicted: {}'.format(misclass_predicted[value]) )
    
plt.show()




