# fichier de test
from unsupervised import LookUpTable, Window, LookUpTrain
import theano
import theano.tensor as T
import numpy as np
from blocks.initialization import IsotropicGaussian
import os
from contextlib import closing
import pickle
from blocks.initialization import Constant, IsotropicGaussian
from supervised import ConvPoolNlp
from active import Architecture


def test_load():
	repo = "data/dico"
	filename = "params_savings_bis34"
	with closing(open(os.path.join(repo, filename), 'rb')) as f:
		data = pickle.load(f)
		import pdb
		pdb.set_trace()
		print 'kikou'

def test_lookUptable():
	# TODO
	lookup_object = LookUpTable(10, 7, weights_init=IsotropicGaussian(0.01))
	lookup_object.initialize()
	x = T.imatrix() # (batch_size, nb_words) batch_size sentences of nb_words
	y = lookup_object.apply(x)
	f = theano.function([x], y, allow_input_downcast=True)
	x_value = np.zeros((3, 12)).astype(int)
	for p in range(3):
		for q in range(12):
			x_value[p,q] = np.random.randint(7)
	result = f(x_value)
	print result.shape
	return
	assert result.shape ==(3, 10, 12)

def test_window():
	n_mot = [6, 7, 8]
	vect_size = [23, 5, 7]
	n_hidden = 14
	dwin = 5
	window = Window(dwin, n_mot, vect_size, n_hidden, n_out=1)
	window.initialize()
	x = T.itensor3()
	y = window.apply(x)
	x_value = np.zeros((2, 3, dwin)).astype(int)
	for p in range(2):
		for i in range(3):
			for q in range(dwin):
				x_value[p,i,q] = np.random.randint(n_mot[i])
	f = theano.function([x], y, allow_input_downcast=True)
	result = f(x_value)
	print result.shape

def test_unsupTraining():
	n_mot = [6, 7, 8]
	vect_size = [23, 5, 7]
	n_hidden = 14
	dwin = 5
	window = LookUpTrain(dwin, n_mot, vect_size, n_hidden, n_out=1)
	window.initialize()
	x = T.itensor3()
	xc = T.itensor3()
	y = window.score(x, xc)
	x_value = np.zeros((2, 3, dwin)).astype(int)
	xc_value = np.zeros((2, 3, dwin)).astype(int)
	for p in range(2):
		for i in range(3):
			for q in range(dwin):
				x_value[p,i,q] = np.random.randint(n_mot[i])
				xc_value[p,i,q] = np.random.randint(n_mot[i])
	f = theano.function([x,xc], y, allow_input_downcast=True)
	result = f(x_value, xc_value)
	print result

def test_cost():
	n_mot = [6, 7, 8]
	vect_size = [23, 5, 7]
	n_hidden = 14
	dwin = 5
	window = LookUpTrain(dwin, n_mot, vect_size, n_hidden, n_out=1)
	window.initialize()
	x = T.itensor3()
	y = T.imatrix()
	cost = window.cost(x, y)
	x_value = np.zeros((2, 3, dwin)).astype(int)
	y_value = np.zeros((2,)).astype(int)
	for p in range(2):
		for i in range(3):
			for q in range(dwin):
				x_value[p,i,q] = np.random.randint(n_mot[i])
		y_value[p] = np.random.randint(2)
	f = theano.function([x,y], cost, allow_input_downcast=True)
	return
	result = f(x_value, xc_value)
	print result


def test_embedding():
	n_mot = [6, 7, 8]
	vect_size = [23, 5, 7]
	n_hidden = 14
	dwin = 5
	window = LookUpTrain(dwin, n_mot, vect_size, n_hidden, n_out=1)
	window.initialize()
	x = T.itensor3()
	x_value = np.zeros((2, 3, dwin)).astype(int)
	for p in range(2):
		for i in range(3):
			for q in range(dwin):
				x_value[p,i,q] = np.random.randint(n_mot[i])
	f = theano.function([x], window.embedding(x), allow_input_downcast=True)		
	print f(x_value).shape

def test_convNLP():
	x = T.matrix('x')
	convNLP = ConvPoolNlp(12, 9, 40, [3, 50], 
			weights_init=IsotropicGaussian(0.01), biases_init=Constant(0.))
	convNLP.initialize()
	y_pred = convNLP.apply(x)
	f = theano.function([x], y_pred, allow_input_downcast=True)
	x_value = np.zeros((12,40))
	y_value = f(x_value)
	print y_value.shape

def test_get_mlp_params():
	n_mot = [6, 7, 8]
	vect_size = [23, 5, 7]
	n_hidden = 14
	dwin = 5
	window = LookUpTrain(dwin, n_mot, vect_size, n_hidden, n_out=1)
	window.initialize()
	window.get_Params()

def test_committee():
	n_mot = [6, 7, 8]
	vect_size = [23, 5, 7]
	n_hidden = 14
	dwin = 5
	window = Architecture(dwin, n_mot, vect_size, n_hidden, n_out=2, dropout=0.5, committee_size=4)
	window.initialize()
	committee = window.generate_committee()

if __name__=="__main__":
	test_committee()
