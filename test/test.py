#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
from contextlib import closing
import os
import pickle

import theano
import theano.tensor as T

from unsupervised import LookUpTrain
from util import get_input_from_files


if __name__ == '__main__':
    
    repo='data/dico'
    output_dico = "embedding_dico_H_S" 
    text_to_test = "Hollande_Test_0.txt"
    
    print "####################"
    print "Chargement du réseau......................",
    t_nlp = LookUpTrain(0, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0], 0) # Default init
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
    
    # Input features
    x = T.itensor3('x') 
    
    # Fonction de prediction : Pour une phrase donnée, quel est le président
    predict = theano.function(inputs=[x], outputs=t_nlp.predict(x), allow_input_downcast=True)
    
    # Qualité de la prédiction
    predict_confidency = theano.function(inputs=[x], outputs=t_nlp.predict_confidency(x)[0], allow_input_downcast=True)
    
    