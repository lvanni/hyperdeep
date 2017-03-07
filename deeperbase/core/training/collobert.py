#!/usr/bin/python
# -*-coding:Utf-8 -*
########################################
# training with Collobert architecture #
########################################
from contextlib import closing
import os
import pickle

from blocks.utils import shared_floatx
import theano

from core.preprocess.dico import get_input_from_files, add_padding
from core.training.lookup import getParams
import numpy as np
import theano.tensor as T

def Adam(cost, params, lr=0.002, b1=0.1, b2=0.001, e=1e-8):
	decay_factor = 1-e
	updates=[]
	updates_reinit = []
	grads=T.grad(cost, params)
	i = shared_floatx(0.,"adam_t")
	i_t = i+1
	updates.append((i,i_t))
	updates_reinit.append((i, 0*i))
	lr = (lr *T.sqrt((1. - (1. - b2)**i_t)) /
		(1. - (1. - b1)**i_t))
	b1_t = 1 - (1 - b1) * decay_factor ** (i_t - 1)
	for p,g in zip(params, grads):
		m = shared_floatx(p.get_value() * 0., "adam_m_"+p.name)
		v = shared_floatx(p.get_value() *0., "adam_v_"+p.name)
		m_t = b1_t*g + (1-b1_t)*g
		v_t = b2*T.sqr(g) + (1-b2)*v
		g_t = m_t/(T.sqrt(v_t)+e)
		updates.append((m,m_t)); updates_reinit.append((m, 0*m))
		updates.append((v,v_t)); updates_reinit.append((v, 0*v))
		updates.append((p, p-lr*g_t))
	
	return updates, updates_reinit

def SGD(cost, params, learning_rate):
	updates =[]
	grad_params = T.grad(cost, params)
	for param, grad_param in zip(params, grad_params):
		update_param = param  - learning_rate*grad_param
		updates.append((param, update_param))
	return updates, None

def RMSProp(cost, params, learning_rate, decay_rate):
	updates =[]
	grad_params = T.grad(cost, params)
	for param, grad_param in zip(params, grad_params):
		caches = shared_floatx(param.get_value() * 0.,"cache_"+param.name)
		update_cache = decay_rate*caches\
					+ (1 - decay_rate)*grad_param**2
		update_param = param  - learning_rate*grad_param/T.sqrt(update_cache + 1e-8)
		updates.append((caches, update_cache))
		updates.append((param, update_param))
	return updates

def build_database(repo, dico_filename, filenames, dwin):
	index = 0
	y_value = []
	x_value = []
	with closing(open(os.path.join(repo, dico_filename), 'rb')) as f:
		dico = pickle.load(f)
	for filename in filenames:
		lines,_ = get_input_from_files(repo, [filename], dico)
		for line in lines:
			x_value.append(line)
			y_value.append(index)
		if index ==0:
			index+=1
	y_value = np.asarray(y_value, dtype=int)

	x_value_0 = [ x_value[i] for i in range(np.argmax(y_value))]
	y_value_0 = [ y_value[i] for i in range(np.argmax(y_value))]
	indexes = np.random.permutation(y_value.shape[0] - np.argmax(y_value))[:np.argmax(y_value)] #TODO PUT IT BACK
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

	paddings = [ [], [], [], []]
	for i in range(dwin/2):
		for i in xrange(4):
			paddings[i].append(dico[i]['PARSING'])
	paddings = np.asarray(paddings)
	#paddings = paddings.reshape((1, paddings.shape[0], paddings.shape[1]))
	x_train_ = [add_padding(elem, paddings).astype(int) for elem in x_train_]
	x_valid_ = [add_padding(elem, paddings) for elem in x_valid_]
	x_test_ = [add_padding(elem, paddings) for elem in x_test_]

	x_train=[]; x_valid=[]; x_test=[]
	y_train=[]; y_valid=[]; y_test=[]
	for elem, label in zip(x_train_, y_train_):
		for i in range(elem.shape[1] -dwin):
			x_train.append(elem[:,i:i+dwin])
			y_train.append(label)
	for elem, label in zip(x_valid_, y_valid_):
		for i in range(elem.shape[1] -dwin):
			x_valid.append(elem[:,i:i+dwin])
			y_valid.append(label)
	for elem, label in zip(x_test_, y_test_):
		for i in range(elem.shape[1] -dwin):
			x_test.append(elem[:,i:i+dwin])
			y_test.append(label)

	index_train = np.random.permutation(len(y_train))
	index_valid = np.random.permutation(len(y_valid))
	index_test = np.random.permutation(len(y_test))
	x_train = [x_train[i].astype(int) for i in index_train]
	x_valid = [x_valid[i].astype(int) for i in index_valid]
	x_test = [x_test[i].astype(int) for i in index_test]
	y_train = [y_train[i] for i in index_train]
	y_valid = [y_valid[i] for i in index_valid]
	y_test = [y_test[i] for i in index_test]

	return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def training_committee_member(instance, learning_rate, decay_rate, train, valid, batch_size):
	x = T.itensor3('x')
	y = T.ivector('y')
	cost = T.mean(instance.cost(x,y))
	error = T.mean(instance.errors(x,y)) # just to pik the right hyperparameters
	params = getParams(instance, x)
	Params = []
	for p in params:
		if p.shape[-1].eval()==2:
			Params.append(p)
	#updates = RMSProp(cost, params, learning_rate, decay_rate)
	if len(Params)!=2:
		print('PROBLEM !')
	updates,_ = Adam(cost, params, learning_rate) # *0.1 ???
	train_model = theano.function(inputs=[x,y], outputs=cost, updates=updates,
					allow_input_downcast=True)

	valid_model = theano.function(inputs=[x, y], outputs=cost, allow_input_downcast=True)
	test_model = theano.function(inputs=[x, y], outputs=error, allow_input_downcast=True)
	x_train, y_train = train
	x_valid, y_valid = valid
	n_train = len(y_train)/batch_size; n_valid = len(y_valid)/batch_size
	increment_rate=0.95
	initial_increment = 3
	increment = initial_increment
	best_valid = np.inf
	valid_cost = []
	#while(increment >0):
	for i in range(2):
		for minibatch_index in range(n_train):
			sentence = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			train_value = train_model(sentence, y_value)
		valid_cost = []
		for minibatch_index in range(n_valid):
			sentence = x_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			y_value = y_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			valid_value = test_model(sentence, y_value)
			valid_cost.append(valid_value)
		"""
		if best_valid*0.95 > np.mean(valid_cost):
			best_valid = np.mean(valid_cost)
			increment = initial_increment
		else:
			increment -=1
		"""
	print (np.mean(valid_cost)*100)
	print ('####')
	return instance
	
def training_committee(committee, learning_rate, decay_rate, train, valid, batch_size):
	committee_new = []
	for instance in committee:
		new_instance = training_committee_member(instance, learning_rate, decay_rate, train, valid, batch_size)
		committee_new.append(new_instance)
	return committee_new
		
