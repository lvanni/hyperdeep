#!/usr/bin/python
# -*-coding:Utf-8 -*
########################################
# training with Collobert architecture #
########################################
from contextlib import closing
import pickle

import theano

from deeperbase.core.config import DWIN, VECT_SIZE, N_HIDDEN, NLP_PATH, LEARNING_RATE, BATCH_SIZE
from deeperbase.core.preprocess.dico import get_input_from_files, add_padding
from deeperbase.core.training.collobert import Adam, SGD
from deeperbase.core.training.lookup import LookUpTrain, getParams
import numpy as np
import theano.tensor as T


def build_confusion_matrix(labels, mistakes):
	# binary output
	confusion = np.zeros((2,2))
	for x,y in zip(labels, mistakes):
		if y==0:
			confusion[x,x]+=1.
		else:
			confusion[x, (x+1)%2] += 1.
	# percentage
	totals = [sum(confusion[0]), sum(confusion[1])]
	confusion[0] /= totals[0]
	confusion[1] /= totals[1]
	print (confusion)

"""
PREPROCESSING : Découpage du texte en segments de 20 mots
Return : 3 échantillions => 1 Training ; 2 Validation ; 3 Test 
"""
def pre_process(corpus, dico, train_percentage=0.8, valid_percentage=0.1, test_percentage=0.1):
	
	assert train_percentage + valid_percentage+test_percentage==1, 'error splitting percentage'
	
	x_value = {}
	y_value = {}

	x_value_ = []
	y_value_ = []

	index = 0
	
	for key, filenames in corpus.iteritems():
		x_value[key] = []
		y_value[key] = []
		for filename in filenames:
			lines = get_input_from_files([filename], dico)
			for line in lines:
				x_value[key].append(line)
				y_value[key].append(index)
				
				x_value_.append(line)
				y_value_.append(index)
		index += 1
	
	nb_max_sentence = max(len(l) for l in y_value.itervalues())

	# Mise à niveau des données globales
	x_value_ = []
	y_value_ = []
	for key in x_value.keys():
		while len(y_value[key]) < nb_max_sentence:
			sentences = x_value[key]
			for line in sentences:
				if len(y_value[key]) < nb_max_sentence:
					x_value[key].append(line)
					y_value[key].append(y_value[key][0])
					
					x_value_.append(line)
					y_value_.append(y_value[key][0])
	for key in x_value.keys():
		print key, len(y_value[key])
	
	y_value_ = np.asarray(y_value_, dtype=int)
	n = len(y_value_)
	
	# decoupage train/valid/test + shuffle
	permutation = np.random.permutation(n)
	n_train = (int) (train_percentage*n)
	n_valid = (int) (valid_percentage*n)
	n_test = (int) (test_percentage*n)		

	x_train_ = [x_value_[permutation[i]] for i in range(n_train) ]
	y_train_ = y_value_[ permutation[:n_train]]

	x_valid_ = [x_value_[permutation[i]] for i in range(n_train, n_train+n_valid) ]
	y_valid_ = y_value_[ permutation[n_train:n_train+n_valid]]

	x_test_ = [x_value_[permutation[i]] for i in range(n_train+n_valid, n) ]
	y_test_ = y_value_[ permutation[n_train+n_valid:n]]			

	"""
	# Mise à niveau des données sur le train
	x_train = {} ; y_train = {}
	x_valid = {} ; y_valid = {}
	x_test = {} ; y_test = {}

	print "Découpage initial :"
	print "\t\tn_train\tn_valid\tn_test"	
	for key, sentences in x_value.iteritems():

		n = len(y_value[key])
		n_train = (int) (train_percentage*n)
		n_valid = (int) (valid_percentage*n)
		n_test = (int) (test_percentage*n)
		
		x_train[key] = x_value[key][:n_train]
		y_train[key] = y_value[key][:n_train]
		x_valid[key] = x_value[key][n_train:n_train+n_valid]
		y_valid[key] = y_value[key][n_train:n_train+n_valid]
		x_test[key] = x_value[key][n_train+n_valid:]
		y_test[key] = y_value[key][n_train+n_valid:]

		print key, "\t", len(y_train[key]), "\t", len(y_valid[key]), "\t", len(y_test[key])	
		#while len(y_train[key]) < 	nb_max_sentence:
		#	for i in xrange(len(x_value[key][:n_train])):
		#		x_train[key] += [x_value[key][:n_train][i]]
		#		y_train[key] += [y_value[key][:n_train][i]]
		#		if len(y_train[key]) >= nb_max_sentence: 
		#			break
		for i in xrange(nb_max_sentence/len(sentences) - 1):
			x_train[key] += x_value[key][:n_train]
			y_train[key] += y_value[key][:n_train]

	x_train_ = [] ; y_train_ = []
	x_valid_ = [] ; y_valid_ = []
	x_test_ = [] ; y_test_ = []
	print "\nAjustement des données :"
	print "\t\tn_train\tn_valid\tn_test"
	for key, sentences in x_value.iteritems():	
		x_train_ += x_train[key]
		y_train_ += y_train[key]
		x_valid_ += x_valid[key]
		y_valid_ += y_valid[key]
		x_test_ += x_test[key]
		y_test_ += y_test[key]
		
		print key, "\t", len(y_train[key]), "\t", len(y_valid[key]), "\t", len(y_test[key])
	
	# decoupage train/valid/test + shuffle
	permutation_train = np.random.permutation(len(x_train_))
	permutation_valid = np.random.permutation(len(x_valid_))
	permutation_test = np.random.permutation(len(x_test_))
	
	x_train_ = [x_train_[permutation_train[i]] for i in range(len(x_train_))]
	y_train_ = [y_train_[permutation_train[i]] for i in range(len(y_train_))]
	
	x_valid_ = [x_valid_[permutation_valid[i]] for i in range(len(x_valid_))]
	y_valid_ = [y_valid_[permutation_valid[i]] for i in range(len(y_valid_))]
	
	x_test_ = [x_test_[permutation_test[i]] for i in range(len(x_test_))]
	y_test_ = [y_test_[permutation_test[i]] for i in range(len(y_test_))]
	
	"""	
	
	# decoupage des phrases en padding
	# padding => pour combler le manque de mots
	paddings = [ [], [], [], []]
	for i in range(DWIN/2):
		for i in xrange(4):
			paddings[i].append(dico[i]['PARSING'])
	paddings = np.asarray(paddings)
	#paddings = paddings.reshape((1, paddings.shape[0], paddings.shape[1]))
	x_train_ = [add_padding(elem, paddings) for elem in x_train_]
	x_valid_ = [add_padding(elem, paddings) for elem in x_valid_]
	x_test_ = [add_padding(elem, paddings) for elem in x_test_]

	# decoupage du texte en segment de 20 mots	
	x_train=[]; x_valid=[]; x_test=[]
	y_train=[]; y_valid=[]; y_test=[]
	for elem, label in zip(x_train_, y_train_):
		for i in range(elem.shape[1] -DWIN):
			x_train.append(elem[:,i:i+DWIN])
			y_train.append(label)
	for elem, label in zip(x_valid_, y_valid_):
		for i in range(elem.shape[1] -DWIN):
			x_valid.append(elem[:,i:i+DWIN])
			y_valid.append(label)
	for elem, label in zip(x_test_, y_test_):
		for i in range(elem.shape[1] -DWIN):
			x_test.append(elem[:,i:i+DWIN])
			y_test.append(label)

	return x_train, x_valid, x_test, y_train, y_valid, y_test

def training(x_train, x_valid, x_test, y_train, y_valid, y_test, dico):
	
	# Nb mot dans le dico/corpus
	n_mot = [len(dico[i]) for i in dico.keys()]
	
	# Natural Langage Processing
	nb_class = np.max(y_train) + 1
	assert np.min(y_train) == 0, ('y_train should contain class labels with the first one indexed at 0 but got %d', np.min(y_train))
	t_nlp = LookUpTrain(DWIN, n_mot, VECT_SIZE, N_HIDDEN, n_out=nb_class)
	t_nlp.initialize()
	#t_nlp.load(repo, filename_load)

	# tensor => Matrice de matrices (i pour entier)
	x = T.itensor3('x') # 3 dimensions : 1) Nombre de phrases en entrée 2) Nombre de mots par phrase 3) Nombre de niveaux => ex Forme/lemme/code/fonction := 4 niveaux 
	y = T.ivector('y')

	cost = T.mean(t_nlp.cost(x, y))
	error = T.mean(t_nlp.errors(x,y))

	params = t_nlp.getParams()

	updates, _ = Adam(cost, params, LEARNING_RATE) # Back Propagation
	#updates, _ = SGD(cost, params, LEARNING_RATE) # Back Propagation

	# Entrainement du modele
	train_model = theano.function(inputs=[x,y], outputs=cost, updates=updates,
					allow_input_downcast=True)

	# Validation 
	valid_model = theano.function(inputs=[x, y], outputs=cost, allow_input_downcast=True)
	
	# Test => Nombre d'err du reseau
	test_model = theano.function(inputs=[x, y], outputs=error, allow_input_downcast=True)

	prediction = theano.function(inputs=[x], outputs=t_nlp.predict(x), allow_input_downcast=True)
	
	"""
	Entrainement
	"""
	batch_size = BATCH_SIZE # BATCH_SIZE phrases à la fois
	n_train = len(y_train)/batch_size
	n_valid = len(y_valid)/batch_size
	n_test = len(y_test)/batch_size
	
	print ("\n##############")
	print ("Start learning")
	print ("##############")

	# number of iterations on the corpus
	epochs = 10 
	best_valid = 200.

	for epoch in range(epochs):
		
		""" TRAINING """
		for minibatch_index in range(n_train):
			sentence = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			#before = valid_model(sentence, y_value)
			train_value = train_model(sentence, y_value)
			if minibatch_index %10==0:
				#after = valid_model(sentence, y_value)
				#print before - after
				""" VALIDATION """
				# Validation => permet de controller la plus value d'un nouvel entrainement
				# On peut arreter l'entrainement si plus aucune evolution.
				valid_cost=[]
				for minibatch_valid in range(n_valid):
					y_value = y_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
					sentence = x_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
					valid_value = test_model(sentence, y_value)
					valid_cost.append(valid_value)
				valid_cost = np.mean(valid_cost)*100
				print ("Valid : " + str(valid_cost))
				if valid_cost < best_valid:
					t_nlp.save() # rajouter option repo et filename pour enregistrer
					best_valid = valid_cost
					print ("saving network...")
				""" TEST """
				"""
				test_cost=[]
				for minibatch_train in range(n_train):
					sentence = x_train[minibatch_train*batch_size:(minibatch_train+1)*batch_size]
					y_value = y_train[minibatch_train*batch_size:(minibatch_train+1)*batch_size]
					test_value = test_model(sentence, y_value)
					test_cost.append(test_value)
				print ("Train : " + str(np.mean(test_cost)*100))
				"""

	print ("DONE.")
	return t_nlp

