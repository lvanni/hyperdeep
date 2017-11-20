#!/usr/bin/python
# -*- coding: utf-8 -*-
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing.label import LabelEncoder

"""
DEEP LEARNING MODEL (COMPILATION)
"""
def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

def create_model(nb_feature, nb_class):

    model = Sequential()
    
    model.add(Dense(10, input_dim=nb_feature, activation='sigmoid'))
    model.add(Dense(nb_class, init='normal', activation='softmax'))
    
    compile_model(model)
    
    return model

"""
TRAINING ALGORITHM
"""
def train(dataset, nb_feature, nb_class):

    X = dataset[:,0:nb_feature].astype(float) # extrait de la 1er colonne à la nième colonne => les features 
    Y = dataset[:,nb_feature] # extrait la nième colonne de la matrice => les classes
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    Y = np_utils.to_categorical(encoded_Y)
    
    model = create_model(nb_feature, nb_class)    

    # Simple validation
    model.fit(X, Y, validation_split=0.1, shuffle=True, nb_epoch=200, batch_size=32, verbose=1)
    
    # kfold validation
    #estimator = KerasClassifier(build_fn=create_model, nb_epoch=512, batch_size=32, verbose=1)
    #fold = KFold(n_splits=10,shuffle=True,random_state=seed)
    #results = cross_val_score(estimator, X, Y, cv=fold)
    #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    return model, encoder.classes_

"""
PREDICT ALGORITHM
"""
def predict(model, dataset, nb_feature, nb_class):
    
    # load data_set
    X = dataset[:,0:nb_feature].astype(float)
    
    # predict
    return np.mean(model.predict(X), axis=0)