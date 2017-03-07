#!/usr/bin/python
# -*-coding:Utf-8 -*
########################################
# training with Collobert architecture #
########################################
from contextlib import closing
import pickle

import theano

from core.config import DWIN, VECT_SIZE, N_HIDDEN, NLP_PATH, LEARNING_RATE, BATCH_SIZE
from core.preprocess.dico import get_input_from_files, add_padding
from core.training.collobert import Adam, SGD
from core.training.lookup import LookUpTrain, getParams
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
	
	y_value = []
	x_value = []
	
	index = 0	
	for key, filenames in corpus.iteritems():
		for filename in filenames:
			lines = get_input_from_files([filename], dico)
			for line in lines:
				x_value.append(line)
				y_value.append(index)
		index += 1
	y_value = np.asarray(y_value, dtype=int)
        n = len(y_value)
	assert train_percentage + valid_percentage+test_percentage==1, 'error splitting percentage'
	
	permutation = np.random.permutation(n)
	n_train = (int) (train_percentage*n)
	n_valid = (int) (valid_percentage*n)
	n_test = (int) (test_percentage*n)

	x_train_ = [ x_value[permutation[i]] for i in range(n_train) ]
	y_train_ = y_value[ permutation[:n_train]]

	x_valid_ = [ x_value[permutation[i]] for i in range(n_train, n_train+n_valid) ]
	y_valid_ = y_value[ permutation[n_train:n_train+n_valid]]

	x_test_ = [ x_value[permutation[i]] for i in range(n_train+n_valid, n) ]
	y_test_ = y_value[ permutation[n_train+n_valid:n]]


	"""
	import pdb; pdb.set_trace()
	
	# balance the samples
	x_value_0 = [ x_value[i] for i in range(np.argmax(y_value)+1)]# put the 0
	y_value_0 = [ y_value[i] for i in range(np.argmax(y_value)+1)]# put the 0
	
	indexes = np.random.permutation(y_value.shape[0] - np.argmax(y_value))[:np.argmax(y_value)]
	x_value_1 = [x_value[i+np.argmax(y_value)] for i in indexes]# balance the numbers
	y_value_1 = [y_value[i+np.argmax(y_value)] for i in indexes]# balance the numbers

	pos_percentage = (int) (len(y_value_0)*0.8)
	neg_percentage = (int) (len(y_value_1)*0.8)
	
	other_pos_percentage = (len(y_value_0) - pos_percentage)/2
	other_neg_percentage = (len(y_value_1) - neg_percentage)/2

	pos_permut = np.random.permutation(len(y_value_0))
	neg_permut = np.random.permutation(len(y_value_1))
	
	x_train = [x_value_0[i] for i in pos_permut[:pos_percentage]] + [x_value_1[i] for i in neg_permut[:neg_percentage]]
	x_valid = [x_value_0[i] for i in pos_permut[pos_percentage:pos_percentage+other_pos_percentage]] + \
		  [x_value_1[i] for i in neg_permut[neg_percentage:neg_percentage+other_neg_percentage]]
	x_test = [x_value_0[i] for i in pos_permut[pos_percentage+other_pos_percentage:]] + \
		  [x_value_1[i] for i in neg_permut[neg_percentage+other_neg_percentage:]]

	y_train = [y_value_0[i] for i in pos_permut[:pos_percentage]] + [y_value_1[i] for i in neg_permut[:neg_percentage]]
	y_valid = [y_value_0[i] for i in pos_permut[pos_percentage:pos_percentage+other_pos_percentage]] + \
		  [y_value_1[i] for i in neg_permut[neg_percentage:neg_percentage+other_neg_percentage]]
	y_test = [y_value_0[i] for i in pos_permut[pos_percentage+other_pos_percentage:]] + \
		  [y_value_1[i] for i in neg_permut[neg_percentage+other_neg_percentage:]]



	index_train = np.random.permutation(len(y_train))
	index_valid = np.random.permutation(len(y_valid))
	index_test = np.random.permutation(len(y_test))

	x_train_ = [x_train[i].astype(int) for i in index_train]
	x_valid_ = [x_valid[i].astype(int) for i in index_valid]
	x_test_ = [x_test[i].astype(int) for i in index_test]
	
	y_train_ = [y_train[i] for i in index_train]
	y_valid_ = [y_valid[i] for i in index_valid]
	y_test_ = [y_test[i] for i in index_test]
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

	"""
	# Melange l'ordre des phrases (pas des mots dans la phrase...)
	index_train = np.random.permutation(len(y_train))
	index_valid = np.random.permutation(len(y_valid))
	index_test = np.random.permutation(len(y_test))
	x_train = [x_train[i].astype(int) for i in index_train]
	x_valid = [x_valid[i].astype(int) for i in index_valid]
	x_test = [x_test[i].astype(int) for i in index_test]
	y_train = [y_train[i] for i in index_train]
	y_valid = [y_valid[i] for i in index_valid]
	y_test = [y_test[i] for i in index_test]
	"""
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

	params = getParams(t_nlp, x)

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
	batch_size = BATCH_SIZE # 32 phrases à la fois
	n_train = len(y_train)/batch_size
	n_valid = len(y_valid)/batch_size
	n_test = len(y_test)/batch_size
	
	print ("\nn_train: ", n_train)
	print ("n_valid: ", n_valid)
	print ("n_test: ", n_test)
	
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

