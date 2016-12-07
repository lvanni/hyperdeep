#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
from contextlib import closing
import os
import pickle
import sys

import theano

import theano.tensor as T
from tcore.training.training_testimport CORPUS
from tcore.training.training_testimport DWIN, VECT_SIZE, N_HIDDEN
from tcore.training.training_testimport pre_process
from ucore.training.unsupervisedimport LookUpTrain
from util import get_input_from_files

if __name__ == '__main__':
    
    text_to_test = sys.argv[1]
    
    print "####################"
    print "Chargement du réseau......................",
    # on charge le dico
    with closing(open(os.path.join(repo, output_dico), 'rb')) as f:
        dico = pickle.load(f)
        
    # Nb mot dans le dico/corpus
    n_mot = [len(dico[i]) for i in dico.keys()]
    
    # Natural Langage Processing
    t_nlp = LookUpTrain(DWIN, n_mot, VECT_SIZE, N_HIDDEN, n_out=2)
    t_nlp.load(repo, "network_state_0")
    print  "OK"
    
    print "####################"
    print "Chargement de l'embedding.................",
    x_value = []
    with closing(open(os.path.join(repo, output_dico), 'rb')) as f:
        dico = pickle.load(f)
    print  "OK"
    
    print "####################"
    print "Chargement de texte.......................",    
    lines, _ = get_input_from_files(repo, [text_to_test], dico)
    for line in lines:
        x_value.append(line)
    print  "OK"
    
    # Preprocessing => découpage du texte
    x_train, x_valid, x_test, y_train, y_valid, y_test = pre_process(repo, output_dico, [text_to_test])
    #print x_train, x_valid, x_test, y_train, y_valid, y_test
    
    # Input features
    x = T.itensor3('x') 
    
    # Fonction de prediction : Pour une phrase donnée, quel est le président
    predict = theano.function(inputs=[x], outputs=t_nlp.predict(x), allow_input_downcast=True)
    
    # Qualité de la prédiction
    predict_confidency = theano.function(inputs=[x], outputs=t_nlp.predict_confidency(x)[0], allow_input_downcast=True)
    
    # test oncorentence
    print "predict :", CORPUS[predict([x_test[0]])].replace(".txt", "")
    
    