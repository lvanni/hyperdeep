########################################
# training with Collobert architecture #
########################################
from supervised import ConvPoolNlp
import theano.tensor as T
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks.cost import MisclassificationRate
from blocks.bricks import Softmax
import theano
import os
from contextlib import closing
import pickle
import numpy as np
from util import build_dictionnary
from unsupervised import LookUpTrain, getParams
from blocks.utils import shared_floatx
from util import get_input_from_files, add_padding


def build_confusion_matrix(labels, mistakes):
	# binary output
	confusion = np.zeros((2,2))
	for x,y in zip(labels, mistakes):
		if y==0:
			confusion[x,x]+=1.
		else:
			confusion[x, (x+1)%2]+=1.
	# percentage
	totals = [sum(confusion[0]), sum(confusion[1])]
	confusion[0] /= totals[0]
	confusion[1] /= totals[1]
	print confusion

def build_lookup(repo, output_dico, filename_load):
	dwin = 9
	with closing(open(os.path.join(repo, output_dico), 'rb')) as f:
		dico = pickle.load(f)
	n_mot = [len(dico[i]) for i in dico.keys()]
	vect_size = [20, 10, 5, 5]
	n_hidden = 25
	x = T.imatrix('x')
	t_nlp = LookUpTrain(dwin, n_mot, vect_size, n_hidden)
	t_nlp.initialize()
	t_nlp.load(repo, filename_load)
	lookup = theano.function(inputs=[x], outputs=t_nlp.embedding(x), allow_input_downcast=True)
	return lookup


def training_Hollande(learning_rate, decay_rate, repo, filenames,
		output_dico, n_out, dwin, vect_size, n_hidden_layer, lookup):
	
	#########
	# MODEL #
	#########
	conv = ConvPoolNlp(n_out, dwin, vect_size, n_hidden_layer,
			weights_init=IsotropicGaussian(0.01), biases_init=Constant(0.))
	conv.initialize()

	#x = T.imatrix()
	x = T.tensor3()
	#x = x.dimshuffle((1,0))
	#z = x.dimshuffle(('x', 0, 1, 2))
	z = x[0]
	y = T.iscalar()
	#y_ = T.imatrix()
	y_pred = conv.apply(z).reshape((1,2))
	cost = Softmax().categorical_cross_entropy(y.reshape((1,)), y_pred).mean()
	error = MisclassificationRate().apply(y.reshape((1,)), y_pred).mean()
	valid_model = theano.function(inputs=[x,y], outputs=cost, allow_input_downcast=True)
	test_model = theano.function(inputs=[x,y], outputs=cost, allow_input_downcast=True)
	predict = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)

	#gradient descent
	params = getParams(conv, z)
	W=[]
	for p, i in zip(params, range(len(params))):
		if p.name[0]=='W':
			W.append(p)
		p.name+='_'+str(i)
	
	# checking metrics for training only
	toto = sum([ w.norm(2) for w in W])
	tata = T.min([w.norm(2) for w in W])
	tutu = T.max([w.norm(2) for w in W])
	print W
	#cost = cost + 1e-4*toto
	
	#calcul du gradient avec RMSProp
	updates = []
	caches = {}
	grad_params = T.grad(cost, params)
	norm_grad = [g.norm(2) for g in grad_params]
	for param, grad_param in zip(params, grad_params):

		if not caches.has_key(param.name):
			caches[param.name] = shared_floatx(param.get_value() * 0.,
												"cache_"+param.name)
		# update rule
		update_momentum = decay_rate*caches[param.name]\
					+learning_rate*grad_param
		updates.append((caches[param.name], update_momentum))
		updates.append((param, update_momentum))
		"""
		update_cache = decay_rate*caches[param.name]\
					+ (1 - decay_rate)*grad_param**2
		update_param = param  - learning_rate*grad_param/T.sqrt(update_cache + 1e-8)
		updates.append((caches[param.name], update_cache))
		updates.append((param, update_param))
		"""
	train_model = theano.function(inputs=[x,y], outputs=cost, updates=updates,
					allow_input_downcast=True)

	observation = theano.function(inputs=[x,y], outputs=norm_grad, allow_input_downcast=True)
	# load data Hollande, Sarkozy
	index = 0
	y_value = []
	x_value = []
	with closing(open(os.path.join(repo, output_dico), 'rb')) as f:
		dico = pickle.load(f)
	for filename in filenames:
		lines = get_input_from_files(repo, [filename], dico)
		for line in lines:
			x_value.append(line)
			y_value.append(index)
		index+=1
	y_value = np.asarray(y_value, dtype=int)
	# balance the samples
	x_value_0 = [ x_value[i] for i in range(np.argmax(y_value))]# put the 0
	y_value_0 = [ y_value[i] for i in range(np.argmax(y_value))]# put the 0
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
	x_test = [x_value_0[i] for i in pos_permut[pos_percentage+other_pos_percentage,:]] + \
		  [x_value_1[i] for i in neg_permut[neg_percentage+other_neg_percentage,:]]

	y_train = [y_value_0[i] for i in pos_permut[:pos_percentage]] + [x_value_1[i] for i in neg_permut[:neg_percentage]]
	y_valid = [y_value_0[i] for i in pos_permut[pos_percentage:pos_percentage+other_pos_percentage]] + \
		  [y_value_1[i] for i in neg_permut[neg_percentage:neg_percentage+other_neg_percentage]]
	y_test = [y_value_0[i] for i in pos_permut[pos_percentage+other_pos_percentage,:]] + \
		  [y_value_1[i] for i in neg_permut[neg_percentage+other_neg_percentage,:]]
	index_train = np.random.permutation(len(y_train))
	index_valid = np.random.permutation(len(y_valid))
	index_test = np.random.permutation(len(y_test))
	x_train = [x_train[i].astype(int) for i in index_train]
	x_valid = [x_valid[i].astype(int) for i in index_valid]
	x_test = [x_test[i].astype(int) for i in index_test]
	y_train = [y_train[i] for i in index_train]
	y_valid = [y_valid[i] for i in index_valid]
	y_test = [y_test[i] for i in index_
	paddings = [ [], [], [], []]
	for i in range(dwin/2):
		for i in xrange(4):
			paddings[i].append(dico[i]['PARSING'])
	paddings = np.asarray(paddings)
	#paddings = paddings.reshape((1, paddings.shape[0], paddings.shape[1]))
	paddings = lookup(paddings)
	x_train = [add_padding(lookup(elem), paddings) for elem in x_train]
	x_valid = [add_padding(lookup(elem), paddings) for elem in x_valid]

	"""
	centroid_x=[ x_train[0], x_train[1]]
	for i in range(10):
		labels_0 = []
		labels_1 = []
		for j in range(len(x_train)):
			dist_0 = sum((x_train[j] - centroid_x[0])**2)
			dist_1 = sum((x_train[j] - centroid_x[1])**2)
			if dist_0 <dist_1:
				labels_0.append(j)
			else:
				labels_1.append(j)

		# recompute data
		centroid_x[0] = sum([x_train[j] for j in labels_0])/(1.0*len(labels_0))	
		centroid_x[1] = sum([x_train[j] for j in labels_1])/(1.0*len(labels_1))

	# print labelling
	labels_0_x=[0,0]
	labels_1_y=[0,0]
	for j in labels_0:
		labels_0_x[y_train[j]]+=1
	for j in labels_1:
		labels_1_y[y_train[j]]+=1
	print labels_0_x
	print labels_1_y
	return
	
	print (len(x_train), len(x_valid))
	"""
	for i in xrange(100):
		index = 0
		for sentence, label in zip(x_train, y_train):
			index +=1
			#sentence_lookup = lookup(sentence)
			#sentence = (sentence - np.mean(sentence))/(np.var(sentence)+1e-8)
			train_model(sentence,label)
			if index %100 ==0:
				print "OBSERVATION : "+str(observation(sentence,label))
		train_cost = []
		for sentence, label in zip(x_train, y_train):
			#sentence_lookup = lookup(sentence)
			#sentence = (sentence - np.mean(sentence))/(np.var(sentence)+1e-8)
			train_cost.append(valid_model(sentence, label))
		print "TRAIN: "+str(np.mean(train_cost)*100)
		mistakes=[]
		for sentence, label in zip(x_valid, y_valid):
			#sentence_lookup = lookup(sentence)
			#sentence_lookup = (sentence - np.mean(sentence))/(np.var(sentence)+1e-8)
			mistakes.append(test_model(sentence, label))
		print "VALID : "+str(np.mean(mistakes)*100)
		#build_confusion_matrix(y_valid, mistakes)


if __name__=='__main__':
	learning_rate = 1e-2
	decay_rate = 0.9
	dwin=12
	repo='data/dico'
	data = 'discours_SarkozyHollande_labelled'
	n_out = 20
	vect_size = 40
	n_hidden_layer=[10]
	filenames = ['HollandeDef.cnr', 'Sarkozy-Inter-DEF.cnr']
	filename_load = 'params_savings_bis_v4_3'
	output_dico="embedding_dico_H_S_v3"
	lookup = build_lookup(repo, output_dico, filename_load)
	training_Hollande(learning_rate, decay_rate, repo, filenames, output_dico,
		n_out, dwin, vect_size, n_hidden_layer, lookup)
